from __future__ import annotations

import json
from typing import Any

from langfence.adapters.base import add_language_instruction, clone_payload
from langfence.constraints import (
    ChoiceConstraint,
    GrammarConstraint,
    JsonSchemaConstraint,
    RegexConstraint,
    StructuralTagConstraint,
)
from langfence.contracts import CompiledRequest, OutputContract, Provider, RequestMode


def compile_vllm_request(
    payload: dict[str, Any],
    contract: OutputContract,
    *,
    mode: RequestMode,
) -> CompiledRequest:
    compiled = clone_payload(payload)
    add_language_instruction(compiled, contract)
    warnings: list[str] = []

    if contract.format is None:
        return CompiledRequest(Provider.VLLM, mode, compiled)

    if mode is RequestMode.OPENAI:
        _compile_openai(compiled, contract, warnings)
    elif mode is RequestMode.NATIVE:
        _compile_native(compiled, contract)
    else:
        raise ValueError(f"Unsupported mode for vLLM: {mode}")

    if contract.language:
        warnings.append(
            "Language policies are validated after generation; vLLM constrained decoding does "
            "not guarantee natural-language semantics."
        )

    return CompiledRequest(Provider.VLLM, mode, compiled, tuple(warnings))


def _compile_openai(
    payload: dict[str, Any],
    contract: OutputContract,
    warnings: list[str],
) -> None:
    constraint = contract.format
    if isinstance(constraint, JsonSchemaConstraint):
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": constraint.name,
                "schema": constraint.schema,
                "strict": constraint.strict,
            },
        }
        if constraint.description:
            payload["response_format"]["json_schema"]["description"] = constraint.description
        return

    if isinstance(constraint, StructuralTagConstraint):
        structured_outputs = _ensure_structured_outputs(payload)
        # vLLM's StructuredOutputsParams.structural_tag expects a JSON-encoded
        # string, not a mapping.
        structured_outputs["structural_tag"] = json.dumps(constraint.spec)
        return

    structured_outputs = _ensure_structured_outputs(payload)
    if isinstance(constraint, RegexConstraint):
        structured_outputs["regex"] = constraint.pattern
    elif isinstance(constraint, ChoiceConstraint):
        structured_outputs["choice"] = list(constraint.choices)
    elif isinstance(constraint, GrammarConstraint):
        structured_outputs["grammar"] = constraint.grammar
        if constraint.syntax != "ebnf":
            warnings.append("vLLM grammar syntax support depends on the configured backend.")
    else:
        raise TypeError(f"Unsupported vLLM constraint: {constraint!r}")


def _compile_native(payload: dict[str, Any], contract: OutputContract) -> None:
    constraint = contract.format
    structured_outputs: dict[str, Any] = {}
    if isinstance(constraint, JsonSchemaConstraint):
        structured_outputs["json"] = constraint.schema
    elif isinstance(constraint, RegexConstraint):
        structured_outputs["regex"] = constraint.pattern
    elif isinstance(constraint, ChoiceConstraint):
        structured_outputs["choice"] = list(constraint.choices)
    elif isinstance(constraint, GrammarConstraint):
        structured_outputs["grammar"] = constraint.grammar
    elif isinstance(constraint, StructuralTagConstraint):
        structured_outputs["structural_tag"] = json.dumps(constraint.spec)
    else:
        raise TypeError(f"Unsupported vLLM constraint: {constraint!r}")

    sampling_params = payload.setdefault("sampling_params", {})
    if not isinstance(sampling_params, dict):
        raise TypeError("payload['sampling_params'] must be a dict for native vLLM mode")
    sampling_params["structured_outputs"] = structured_outputs


def _ensure_structured_outputs(payload: dict[str, Any]) -> dict[str, Any]:
    extra_body = payload.setdefault("extra_body", {})
    if not isinstance(extra_body, dict):
        raise TypeError("payload['extra_body'] must be a dict")

    structured_outputs = extra_body.setdefault("structured_outputs", {})
    if not isinstance(structured_outputs, dict):
        raise TypeError("payload['extra_body']['structured_outputs'] must be a dict")
    return structured_outputs
