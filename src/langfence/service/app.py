from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request

from langfence.adapters import compile_request
from langfence.constraints import GrammarConstraint, StructuralTagConstraint
from langfence.contracts import OutputContract
from langfence.privacy import REDACTED, redact_for_display
from langfence.serialization import contract_from_dict
from langfence.service.schemas import CompileRequestBody, ValidateRequestBody
from langfence.validation import (
    ValidationIssue,
    ValidationResult,
    validate_output,
    validate_provider_enforced_output,
)


def create_app(
    *,
    provider: str,
    base_url: str,
    default_contract: OutputContract | None = None,
    include_provider_error_body: bool = False,
) -> FastAPI:
    app = FastAPI(title="LangFence proxy", version="0.1.1")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/compile")
    async def compile_endpoint(body: CompileRequestBody) -> dict[str, Any]:
        contract = contract_from_dict(body.contract)
        compiled = compile_request(
            provider=body.provider,
            messages=body.messages,
            contract=contract,
            mode=body.mode,
            base_payload=body.base_payload,
        )
        return {
            "provider": compiled.provider.value,
            "mode": compiled.mode.value,
            "payload": redact_for_display(compiled.payload) if body.redact else compiled.payload,
            "redacted": body.redact,
            "warnings": list(compiled.warnings),
        }

    @app.post("/validate")
    async def validate_endpoint(body: ValidateRequestBody) -> dict[str, Any]:
        result = validate_output(body.output, contract_from_dict(body.contract))
        payload = _validation_payload(result.issues, result.ok)
        payload["parsed"] = REDACTED if body.redact and result.parsed is not None else result.parsed
        payload["redacted"] = body.redact
        return payload

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

        contract = _extract_contract(body, default_contract)
        if contract is None:
            raise HTTPException(
                status_code=400,
                detail="Missing output contract. Pass x-output-contract in the request body or "
                "start the proxy with a default contract.",
            )

        messages = body.pop("messages", [])
        body.pop("x-output-contract", None)
        compiled = compile_request(
            provider=provider,
            messages=messages,
            contract=contract,
            base_payload=body,
        )
        headers = {}
        if authorization:
            headers["Authorization"] = authorization

        timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                base_url.rstrip("/") + "/chat/completions",
                json=compiled.payload,
                headers=headers,
            )
        if response.status_code >= 400:
            detail: dict[str, Any] = {
                "message": "Provider returned an error.",
                "provider_status_code": response.status_code,
            }
            if include_provider_error_body:
                detail["provider_error_body"] = response.text
            raise HTTPException(status_code=response.status_code, detail=detail)

        try:
            response_data = response.json()
        except ValueError as exc:
            raise HTTPException(
                status_code=502,
                detail="Provider returned a non-JSON body",
            ) from exc
        if not isinstance(response_data, dict):
            raise HTTPException(status_code=502, detail="Provider returned a non-object JSON body")
        data: dict[str, Any] = response_data
        text = _extract_openai_text(data)
        validation = _validate_provider_output(provider, text, contract)
        data["output_contract"] = _validation_payload(validation.issues, validation.ok)
        return data

    return app


def _extract_contract(
    body: dict[str, Any],
    default_contract: OutputContract | None,
) -> OutputContract | None:
    contract_data = body.get("x-output-contract")
    if isinstance(contract_data, dict):
        return contract_from_dict(contract_data)
    return default_contract


def _extract_openai_text(data: dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Provider response is missing message content.",
        ) from exc
    if isinstance(content, str):
        return content
    raise HTTPException(status_code=502, detail="Provider response content must be a string.")


def _validation_payload(issues: tuple[ValidationIssue, ...], ok: bool) -> dict[str, Any]:
    return {
        "ok": ok,
        "issues": [
            {
                "code": issue.code,
                "message": issue.message,
                "severity": issue.severity,
                "path": issue.path,
                "metadata": issue.metadata,
            }
            for issue in issues
        ],
    }


def _validate_provider_output(
    provider: str, text: str, contract: OutputContract
) -> ValidationResult:
    if provider in {"vllm", "sglang"} and isinstance(
        contract.format,
        GrammarConstraint | StructuralTagConstraint,
    ):
        return validate_provider_enforced_output(text, contract)
    return validate_output(text, contract)
