from __future__ import annotations

import json
from typing import Any

import pytest

from langfence import (
    ChoiceConstraint,
    GrammarConstraint,
    JsonSchemaConstraint,
    OutputContract,
    RegexConstraint,
    StructuralTagConstraint,
    compile_request,
)
from langfence.constraints import OutputConstraint

_MESSAGES = [{"role": "user", "content": "classify"}]
_BASE = {"model": "m"}
_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def _constraint(kind: str) -> OutputConstraint:
    if kind == "json_schema":
        return JsonSchemaConstraint(schema=_SCHEMA, name="answer")
    if kind == "regex":
        return RegexConstraint(r"\d{3}")
    if kind == "choice":
        return ChoiceConstraint(["approved", "rejected"])
    if kind == "grammar":
        return GrammarConstraint('root ::= "ok"', syntax="ebnf")
    if kind == "structural_tag":
        return StructuralTagConstraint(spec={"begin": "<a>"})
    raise AssertionError(f"unknown constraint kind: {kind}")


def _compile(provider: str, kind: str) -> dict[str, Any]:
    compiled = compile_request(
        provider,
        _MESSAGES,
        OutputContract(format=_constraint(kind)),
        base_payload=dict(_BASE),
    )
    return compiled.payload


def _assert_snapshot(provider: str, kind: str, expected: dict[str, Any]) -> None:
    payload = _compile(provider, kind)
    assert json.dumps(payload, sort_keys=True) == json.dumps(expected, sort_keys=True)


# --- vLLM ---------------------------------------------------------------------


def test_sglang_choice_snapshot() -> None:
    compiled = compile_request(
        "sglang",
        [{"role": "user", "content": "classify"}],
        OutputContract(format=ChoiceConstraint(["approved", "rejected"])),
        base_payload={"model": "m"},
    )

    assert json.dumps(compiled.payload, sort_keys=True) == json.dumps(
        {
            "extra_body": {"regex": "(?:approved|rejected)"},
            "messages": [{"role": "user", "content": "classify"}],
            "model": "m",
        },
        sort_keys=True,
    )


_JSON_SCHEMA_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "answer", "schema": _SCHEMA, "strict": True},
}


def test_vllm_json_schema_snapshot() -> None:
    _assert_snapshot(
        "vllm",
        "json_schema",
        {
            "model": "m",
            "messages": _MESSAGES,
            "response_format": _JSON_SCHEMA_RESPONSE_FORMAT,
        },
    )


def test_vllm_regex_snapshot() -> None:
    _assert_snapshot(
        "vllm",
        "regex",
        {
            "model": "m",
            "messages": _MESSAGES,
            "extra_body": {"structured_outputs": {"regex": r"\d{3}"}},
        },
    )


def test_vllm_choice_snapshot() -> None:
    _assert_snapshot(
        "vllm",
        "choice",
        {
            "model": "m",
            "messages": _MESSAGES,
            "extra_body": {"structured_outputs": {"choice": ["approved", "rejected"]}},
        },
    )


def test_vllm_grammar_snapshot() -> None:
    _assert_snapshot(
        "vllm",
        "grammar",
        {
            "model": "m",
            "messages": _MESSAGES,
            "extra_body": {"structured_outputs": {"grammar": 'root ::= "ok"'}},
        },
    )


def test_vllm_structural_tag_snapshot() -> None:
    # vLLM expects structural_tag as a JSON-encoded string inside structured_outputs.
    _assert_snapshot(
        "vllm",
        "structural_tag",
        {
            "model": "m",
            "messages": _MESSAGES,
            "extra_body": {"structured_outputs": {"structural_tag": json.dumps({"begin": "<a>"})}},
        },
    )


# --- SGLang -------------------------------------------------------------------


def test_sglang_json_schema_snapshot() -> None:
    _assert_snapshot(
        "sglang",
        "json_schema",
        {
            "model": "m",
            "messages": _MESSAGES,
            "response_format": _JSON_SCHEMA_RESPONSE_FORMAT,
        },
    )


def test_sglang_regex_snapshot() -> None:
    _assert_snapshot(
        "sglang",
        "regex",
        {
            "model": "m",
            "messages": _MESSAGES,
            "extra_body": {"regex": r"\d{3}"},
        },
    )


def test_sglang_choice_snapshot_is_unanchored() -> None:
    _assert_snapshot(
        "sglang",
        "choice",
        {
            "model": "m",
            "messages": _MESSAGES,
            "extra_body": {"regex": "(?:approved|rejected)"},
        },
    )


def test_sglang_grammar_ebnf_snapshot() -> None:
    _assert_snapshot(
        "sglang",
        "grammar",
        {
            "model": "m",
            "messages": _MESSAGES,
            "extra_body": {"ebnf": 'root ::= "ok"'},
        },
    )


def test_sglang_structural_tag_snapshot() -> None:
    _assert_snapshot(
        "sglang",
        "structural_tag",
        {
            "model": "m",
            "messages": _MESSAGES,
            "response_format": {"type": "structural_tag", "begin": "<a>"},
        },
    )


def test_sglang_grammar_lark_raises() -> None:
    with pytest.raises(ValueError, match="EBNF"):
        compile_request(
            "sglang",
            _MESSAGES,
            OutputContract(format=GrammarConstraint('root ::= "ok"', syntax="lark")),
            base_payload=dict(_BASE),
        )


# --- OpenAI-compatible --------------------------------------------------------


def test_openai_compatible_json_schema_snapshot() -> None:
    _assert_snapshot(
        "openai-compatible",
        "json_schema",
        {
            "model": "m",
            "messages": _MESSAGES,
            "response_format": _JSON_SCHEMA_RESPONSE_FORMAT,
        },
    )


def test_openai_compatible_regex_snapshot() -> None:
    _assert_snapshot(
        "openai-compatible",
        "regex",
        {
            "model": "m",
            "messages": [
                {
                    "role": "system",
                    "content": "Return only text that fully matches this regex: \\d{3}",
                },
                *_MESSAGES,
            ],
        },
    )


def test_openai_compatible_choice_snapshot() -> None:
    _assert_snapshot(
        "openai-compatible",
        "choice",
        {
            "model": "m",
            "messages": [
                {
                    "role": "system",
                    "content": "Return exactly one of these values: approved, rejected",
                },
                *_MESSAGES,
            ],
        },
    )


def test_openai_compatible_grammar_snapshot() -> None:
    _assert_snapshot(
        "openai-compatible",
        "grammar",
        {
            "model": "m",
            "messages": [
                {
                    "role": "system",
                    "content": 'Return only text that conforms to this grammar: root ::= "ok"',
                },
                *_MESSAGES,
            ],
        },
    )


def test_openai_compatible_structural_tag_snapshot() -> None:
    _assert_snapshot(
        "openai-compatible",
        "structural_tag",
        {
            "model": "m",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return output using only this structural tag specification: "
                        '{"begin":"<a>"}'
                    ),
                },
                *_MESSAGES,
            ],
        },
    )


# --- LiteLLM ------------------------------------------------------------------


def test_litellm_json_schema_snapshot() -> None:
    _assert_snapshot(
        "litellm",
        "json_schema",
        {
            "model": "m",
            "messages": _MESSAGES,
            "response_format": _JSON_SCHEMA_RESPONSE_FORMAT,
        },
    )


def test_litellm_regex_snapshot() -> None:
    _assert_snapshot(
        "litellm",
        "regex",
        {
            "model": "m",
            "messages": [
                {
                    "role": "system",
                    "content": "Return only text that fully matches this regex: \\d{3}",
                },
                *_MESSAGES,
            ],
        },
    )


def test_litellm_choice_snapshot() -> None:
    _assert_snapshot(
        "litellm",
        "choice",
        {
            "model": "m",
            "messages": [
                {
                    "role": "system",
                    "content": "Return exactly one of these values: approved, rejected",
                },
                *_MESSAGES,
            ],
        },
    )


def test_litellm_grammar_snapshot() -> None:
    _assert_snapshot(
        "litellm",
        "grammar",
        {
            "model": "m",
            "messages": [
                {
                    "role": "system",
                    "content": 'Return only text that conforms to this grammar: root ::= "ok"',
                },
                *_MESSAGES,
            ],
        },
    )


def test_litellm_structural_tag_snapshot() -> None:
    _assert_snapshot(
        "litellm",
        "structural_tag",
        {
            "model": "m",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return output using only this structural tag specification: "
                        '{"begin":"<a>"}'
                    ),
                },
                *_MESSAGES,
            ],
        },
    )


# --- Anthropic-compatible -----------------------------------------------------


def test_anthropic_compatible_json_schema_snapshot() -> None:
    _assert_snapshot(
        "anthropic-compatible",
        "json_schema",
        {
            "model": "m",
            "messages": _MESSAGES,
            "max_tokens": 1024,
            "system": (
                "Return only valid JSON matching this JSON Schema. Schema name: answer. "
                'Schema: {"type":"object","properties":{"answer":{"type":"string"}},'
                '"required":["answer"]}'
            ),
        },
    )


def test_anthropic_compatible_regex_snapshot() -> None:
    _assert_snapshot(
        "anthropic-compatible",
        "regex",
        {
            "model": "m",
            "messages": _MESSAGES,
            "max_tokens": 1024,
            "system": "Return only text that fully matches this regex: \\d{3}",
        },
    )


def test_anthropic_compatible_choice_snapshot() -> None:
    _assert_snapshot(
        "anthropic-compatible",
        "choice",
        {
            "model": "m",
            "messages": _MESSAGES,
            "max_tokens": 1024,
            "system": "Return exactly one of these values: approved, rejected",
        },
    )


def test_anthropic_compatible_grammar_snapshot() -> None:
    _assert_snapshot(
        "anthropic-compatible",
        "grammar",
        {
            "model": "m",
            "messages": _MESSAGES,
            "max_tokens": 1024,
            "system": 'Return only text that conforms to this grammar: root ::= "ok"',
        },
    )


def test_anthropic_compatible_structural_tag_snapshot() -> None:
    _assert_snapshot(
        "anthropic-compatible",
        "structural_tag",
        {
            "model": "m",
            "messages": _MESSAGES,
            "max_tokens": 1024,
            "system": (
                "Return output using only this structural tag specification: " '{"begin":"<a>"}'
            ),
        },
    )
