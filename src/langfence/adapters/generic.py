from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langfence.adapters.base import add_language_instruction, clone_payload
from langfence.constraints import (
    ChoiceConstraint,
    GrammarConstraint,
    JsonSchemaConstraint,
    OutputConstraint,
    RegexConstraint,
    StructuralTagConstraint,
)
from langfence.contracts import CompiledRequest, OutputContract, Provider, RequestMode
from langfence.language import language_instruction


def compile_openai_compatible_request(
    payload: dict[str, Any],
    contract: OutputContract,
    *,
    provider: Provider,
    mode: RequestMode,
) -> CompiledRequest:
    if mode is not RequestMode.OPENAI:
        raise ValueError(f"{provider.value} supports only openai request mode")

    compiled = clone_payload(payload)
    add_language_instruction(compiled, contract)
    warnings: list[str] = []

    constraint = contract.format
    if isinstance(constraint, JsonSchemaConstraint):
        compiled["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": constraint.name,
                "schema": constraint.schema,
                "strict": constraint.strict,
            },
        }
        if constraint.description:
            compiled["response_format"]["json_schema"]["description"] = constraint.description
    elif constraint is not None:
        instruction = format_instruction(constraint)
        if instruction:
            _insert_system_instruction(compiled, instruction)
        warnings.append(
            f"{provider.value} has no portable request field for {constraint.kind}; "
            "format enforcement is post-validation only."
        )

    if contract.language:
        warnings.append(
            f"{provider.value} language policies are prompt guidance plus post-validation."
        )

    return CompiledRequest(provider, mode, compiled, tuple(warnings))


def compile_anthropic_request(
    payload: dict[str, Any],
    contract: OutputContract,
    *,
    provider: Provider,
    mode: RequestMode,
) -> CompiledRequest:
    if mode is not RequestMode.ANTHROPIC:
        raise ValueError(f"{provider.value} supports only anthropic request mode")

    compiled = clone_payload(payload)
    messages = compiled.pop("messages", [])
    if not isinstance(messages, list):
        raise TypeError("payload['messages'] must be a list for Anthropic-compatible mode")

    system_parts: list[str] = []
    existing_system = compiled.pop("system", None)
    if existing_system is not None:
        system_parts.append(_content_to_text(existing_system))

    anthropic_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise TypeError("Anthropic-compatible messages must be mappings")
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if role in {"system", "developer"}:
            system_parts.append(_content_to_text(content))
            continue
        if role not in {"user", "assistant"}:
            raise ValueError(
                "Anthropic-compatible messages support only system, developer, user, "
                f"and assistant roles; got {role!r}."
            )
        anthropic_messages.append({"role": role, "content": content})

    if contract.prompt_instruction:
        system_parts.append(contract.prompt_instruction)
    if contract.format is not None:
        system_parts.append(format_instruction(contract.format))
    if contract.language is not None:
        instruction = language_instruction(contract.language)
        if instruction:
            system_parts.append(instruction)

    compiled["messages"] = anthropic_messages
    compiled.setdefault("max_tokens", 1024)
    if system_parts:
        compiled["system"] = "\n\n".join(part for part in system_parts if part)

    warnings: list[str] = []
    if contract.format is not None:
        warnings.append(
            f"{provider.value} has no portable constrained-decoding request field; "
            "format enforcement is post-validation only."
        )
    if contract.language is not None:
        warnings.append(
            f"{provider.value} language policies are prompt guidance plus post-validation."
        )
    return CompiledRequest(provider, RequestMode.ANTHROPIC, compiled, tuple(warnings))


def format_instruction(constraint: OutputConstraint) -> str:
    if isinstance(constraint, JsonSchemaConstraint):
        schema = json.dumps(constraint.schema, ensure_ascii=False, separators=(",", ":"))
        return (
            "Return only valid JSON matching this JSON Schema. "
            f"Schema name: {constraint.name}. Schema: {schema}"
        )
    if isinstance(constraint, RegexConstraint):
        return f"Return only text that fully matches this regex: {constraint.pattern}"
    if isinstance(constraint, ChoiceConstraint):
        return "Return exactly one of these values: " + ", ".join(constraint.choices)
    if isinstance(constraint, GrammarConstraint):
        return f"Return only text that conforms to this grammar: {constraint.grammar}"
    if isinstance(constraint, StructuralTagConstraint):
        spec = json.dumps(constraint.spec, ensure_ascii=False, separators=(",", ":"))
        return f"Return output using only this structural tag specification: {spec}"
    return ""


def _insert_system_instruction(payload: dict[str, Any], instruction: str) -> None:
    messages = payload.setdefault("messages", [])
    if not isinstance(messages, list):
        raise TypeError("payload['messages'] must be a list")
    messages.insert(0, {"role": "system", "content": instruction})


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)
