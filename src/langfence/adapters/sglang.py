from __future__ import annotations

import json
import re
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


def compile_sglang_request(
    payload: dict[str, Any],
    contract: OutputContract,
    *,
    mode: RequestMode,
) -> CompiledRequest:
    compiled = clone_payload(payload)
    add_language_instruction(compiled, contract)
    warnings: list[str] = []

    if contract.format is None:
        return CompiledRequest(Provider.SGLANG, mode, compiled)

    if mode is RequestMode.OPENAI:
        _compile_openai(compiled, contract, warnings)
    elif mode is RequestMode.NATIVE:
        _compile_native(compiled, contract, warnings)
    else:
        raise ValueError(f"Unsupported mode for SGLang: {mode}")

    if contract.language:
        warnings.append(
            "Language policies are validated after generation; SGLang constrained decoding does "
            "not guarantee natural-language semantics."
        )

    return CompiledRequest(Provider.SGLANG, mode, compiled, tuple(warnings))


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
    elif isinstance(constraint, RegexConstraint):
        _ensure_extra_body(payload)["regex"] = constraint.pattern
    elif isinstance(constraint, ChoiceConstraint):
        _ensure_extra_body(payload)["regex"] = choices_to_regex(constraint.choices)
        warnings.append("SGLang has no stable OpenAI choice field; compiled choices to regex.")
    elif isinstance(constraint, GrammarConstraint):
        _require_ebnf(constraint)
        _ensure_extra_body(payload)["ebnf"] = constraint.grammar
    elif isinstance(constraint, StructuralTagConstraint):
        payload["response_format"] = {"type": "structural_tag", **constraint.spec}
    else:
        raise TypeError(f"Unsupported SGLang constraint: {constraint!r}")


def _compile_native(
    payload: dict[str, Any],
    contract: OutputContract,
    warnings: list[str],
) -> None:
    constraint = contract.format
    sampling_params = payload.setdefault("sampling_params", {})
    if not isinstance(sampling_params, dict):
        raise TypeError("payload['sampling_params'] must be a dict for native SGLang mode")

    if isinstance(constraint, JsonSchemaConstraint):
        sampling_params["json_schema"] = json.dumps(constraint.schema)
    elif isinstance(constraint, RegexConstraint):
        sampling_params["regex"] = constraint.pattern
    elif isinstance(constraint, ChoiceConstraint):
        sampling_params["regex"] = choices_to_regex(constraint.choices)
        warnings.append("SGLang has no stable native choice field; compiled choices to regex.")
    elif isinstance(constraint, GrammarConstraint):
        _require_ebnf(constraint)
        sampling_params["ebnf"] = constraint.grammar
    elif isinstance(constraint, StructuralTagConstraint):
        sampling_params["structural_tag"] = json.dumps(constraint.spec)
    else:
        raise TypeError(f"Unsupported SGLang constraint: {constraint!r}")


def _ensure_extra_body(payload: dict[str, Any]) -> dict[str, Any]:
    extra_body = payload.setdefault("extra_body", {})
    if not isinstance(extra_body, dict):
        raise TypeError("payload['extra_body'] must be a dict")
    return extra_body


def _require_ebnf(constraint: GrammarConstraint) -> None:
    if constraint.syntax != "ebnf":
        raise ValueError(
            f"SGLang supports only EBNF grammars; got syntax={constraint.syntax!r}. "
            "Convert the grammar to EBNF or target a provider that supports it."
        )


def choices_to_regex(choices: tuple[str, ...]) -> str:
    if not choices:
        raise ValueError("ChoiceConstraint requires at least one choice")
    # Constrained-decoding regex backends match the full output implicitly and
    # several reject ^/$ anchors, so emit an unanchored alternation.
    return "(?:" + "|".join(re.escape(choice) for choice in choices) + ")"
