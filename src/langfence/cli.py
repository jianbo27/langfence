from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import click
import httpx
import typer

from langfence.adapters import compile_request
from langfence.clients import LangFenceClient, LangFenceClientError, LangFenceHTTPError
from langfence.contracts import RequestMode
from langfence.privacy import REDACTED, redact_for_display
from langfence.serialization import load_contract
from langfence.validation import ValidationIssue, ValidationResult, validate_output

app = typer.Typer(no_args_is_help=True, help="Compile and validate LangFences.")

CHAT_PROVIDERS = {
    "vllm",
    "sglang",
    "openai",
    "openai-compatible",
    "litellm",
    "anthropic",
    "anthropic-compatible",
}
ANTHROPIC_VERSION = "2023-06-01"


@app.command()
def compile(
    provider: Annotated[
        str,
        typer.Option(
            help=(
                "Provider: vllm, sglang, openai-compatible, litellm, "
                "anthropic-compatible."
            )
        ),
    ],
    contract: Annotated[Path, typer.Option(help="YAML contract file.")],
    mode: Annotated[str, typer.Option(help="Request mode: openai, anthropic, or native.")] = (
        "openai"
    ),
    base_payload: Annotated[
        Path | None,
        typer.Option(help="Optional JSON file with an existing request payload."),
    ] = None,
    show_sensitive: Annotated[
        bool,
        typer.Option(help="Print prompt/output content and secrets instead of redacting them."),
    ] = False,
) -> None:
    loaded_contract = load_contract(contract)
    payload: dict[str, Any] = {}
    if base_payload:
        payload = json.loads(base_payload.read_text())
    compiled = compile_request(
        provider=provider,
        messages=payload.pop("messages", []),
        contract=loaded_contract,
        mode=RequestMode(mode),
        base_payload=payload,
    )
    typer.echo(
        json.dumps(
            {
                "provider": compiled.provider.value,
                "mode": compiled.mode.value,
                "payload": (
                    compiled.payload if show_sensitive else redact_for_display(compiled.payload)
                ),
                "redacted": not show_sensitive,
                "warnings": list(compiled.warnings),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command()
def validate(
    contract: Annotated[Path, typer.Option(help="YAML contract file.")],
    input: Annotated[Path, typer.Option(help="Text file containing model output.")],
    show_sensitive: Annotated[
        bool,
        typer.Option(help="Print parsed model output instead of redacting it."),
    ] = False,
) -> None:
    loaded_contract = load_contract(contract)
    result = validate_output(input.read_text(), loaded_contract)
    typer.echo(
        json.dumps(
            {
                "ok": result.ok,
                "issues": [_issue_to_dict(issue) for issue in result.issues],
                "parsed": result.parsed if show_sensitive else REDACTED if result.parsed else None,
                "redacted": not show_sensitive,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not result.ok:
        raise typer.Exit(1)


@app.command()
def chat(
    provider: Annotated[
        str,
        typer.Option(
            help=(
                "Provider: vllm, sglang, openai, openai-compatible, litellm, "
                "anthropic, or anthropic-compatible."
            )
        ),
    ],
    base_url: Annotated[str, typer.Option(help="Provider API base URL.")],
    model: Annotated[str, typer.Option(help="Model name to request.")],
    contract: Annotated[Path, typer.Option(help="YAML contract file.")],
    prompt: Annotated[str | None, typer.Option(help="Single user prompt text.")] = None,
    messages: Annotated[
        str | None,
        typer.Option(help="JSON messages array or path to a JSON file with messages."),
    ] = None,
    api_key_env: Annotated[
        str | None,
        typer.Option(help="Environment variable containing the API key."),
    ] = None,
    max_retries: Annotated[
        int,
        typer.Option(help="Retry count after invalid model output."),
    ] = 0,
    show_sensitive: Annotated[
        bool,
        typer.Option(help="Print raw model output text."),
    ] = False,
) -> None:
    normalized_provider = _normalize_chat_provider(provider)
    if max_retries < 0:
        raise typer.BadParameter("--max-retries must be greater than or equal to 0.")

    loaded_contract = load_contract(contract)
    chat_messages = _resolve_chat_messages(prompt=prompt, messages=messages)
    api_key = _api_key_from_env(api_key_env)

    try:
        with httpx.Client(timeout=60.0) as http_client:
            client = LangFenceClient(
                provider=normalized_provider,
                base_url=base_url,
                model=model,
                contract=loaded_contract,
                max_retries=max_retries,
                api_key=api_key,
                client=http_client,
            )
            result = client.chat(chat_messages)
    except (LangFenceClientError, httpx.HTTPError, ValueError) as exc:
        raise click.ClickException(_request_error_message(exc)) from exc

    output: dict[str, Any] = {
        "provider": normalized_provider,
        "profile": result.profile,
        "transport": result.transport,
        "model": model,
        "attempts": result.attempts,
        "validation": _validation_summary(result.validation),
        "redacted": not show_sensitive,
        "warnings": list(result.warnings),
    }
    if show_sensitive:
        output["text"] = result.text

    typer.echo(json.dumps(output, ensure_ascii=False, indent=2))
    if not result.ok:
        raise typer.Exit(1)


@app.command()
def proxy(
    provider: Annotated[str, typer.Option(help="Provider: vllm or sglang.")],
    base_url: Annotated[str, typer.Option(help="Provider OpenAI-compatible base URL.")],
    contract: Annotated[Path | None, typer.Option(help="Default YAML contract file.")] = None,
    host: Annotated[str, typer.Option(help="Proxy bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Proxy bind port.")] = 8090,
    include_provider_error_body: Annotated[
        bool,
        typer.Option(help="Return raw provider error bodies. Disabled by default to avoid leaks."),
    ] = False,
) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter("Install with the 'service' extra to run the proxy.") from exc

    from langfence.service.app import create_app

    default_contract = load_contract(contract) if contract else None
    uvicorn.run(
        create_app(
            provider=provider,
            base_url=base_url,
            default_contract=default_contract,
            include_provider_error_body=include_provider_error_body,
        ),
        host=host,
        port=port,
    )


def _normalize_chat_provider(provider: str) -> str:
    normalized = provider.strip().lower().replace("_", "-")
    if normalized not in CHAT_PROVIDERS:
        allowed = ", ".join(sorted(CHAT_PROVIDERS))
        raise typer.BadParameter(
            f"Unsupported chat provider: {provider}. Expected one of: {allowed}."
        )
    return normalized


def _resolve_chat_messages(
    *,
    prompt: str | None,
    messages: str | None,
) -> list[dict[str, Any]]:
    if prompt and messages:
        raise typer.BadParameter("Use either --prompt or --messages, not both.")
    if prompt is None and messages is None:
        raise typer.BadParameter("Provide either --prompt TEXT or --messages JSON_OR_PATH.")
    if prompt is not None:
        return [{"role": "user", "content": prompt}]
    if messages is None:
        raise typer.BadParameter("Provide either --prompt TEXT or --messages JSON_OR_PATH.")

    data = _load_messages_data(messages)
    if isinstance(data, Mapping) and "messages" in data:
        data = data["messages"]
    if not isinstance(data, list):
        raise typer.BadParameter("--messages must contain a JSON array of message objects.")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, Mapping):
            raise typer.BadParameter(f"--messages item {index} must be a JSON object.")
        role = item.get("role")
        if not isinstance(role, str) or not role:
            raise typer.BadParameter(f"--messages item {index} must include a string role.")
        if "content" not in item:
            raise typer.BadParameter(f"--messages item {index} must include content.")
        normalized.append(dict(item))
    return normalized


def _load_messages_data(raw: str) -> Any:
    stripped = raw.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"--messages is not valid JSON: {exc.msg}.") from exc

    path = Path(raw)
    if not path.exists():
        raise typer.BadParameter("--messages must be valid JSON or a path to a JSON file.")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"--messages file is not valid JSON: {exc.msg}.") from exc


def _api_key_from_env(api_key_env: str | None) -> str | None:
    if api_key_env is None:
        return None

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise typer.BadParameter(f"Environment variable {api_key_env!r} is not set.")
    return api_key


def _validation_summary(result: ValidationResult) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "issue_count": len(result.issues),
        "error_count": len(result.errors),
        "warning_count": len(result.warnings),
        "issues": [_issue_to_dict(issue) for issue in result.issues],
    }


def _issue_to_dict(issue: ValidationIssue) -> dict[str, Any]:
    return {
        "code": issue.code,
        "message": issue.message,
        "severity": issue.severity,
        "path": issue.path,
        "metadata": issue.metadata,
    }


def _request_error_message(error: Exception) -> str:
    if isinstance(error, LangFenceHTTPError):
        return f"Provider request failed with HTTP {error.status_code}."
    if isinstance(error, httpx.RequestError):
        return f"Provider request failed: {error.__class__.__name__}."
    return str(error)


if __name__ == "__main__":
    app()
