from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from langfence import (
    ChoiceConstraint,
    JsonSchemaConstraint,
    LangFence,
    LanguagePolicy,
    OutputContract,
)
from langfence.clients import LangFenceClient

_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def _openai_response(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_validate_accepts_conforming_output() -> None:
    fence = LangFence(format=JsonSchemaConstraint(schema=_SCHEMA))

    result = fence.validate('{"answer": "ok"}')

    assert result.ok
    assert result.parsed == {"answer": "ok"}


def test_is_valid_reports_format_violation() -> None:
    fence = LangFence(format=JsonSchemaConstraint(schema=_SCHEMA))

    assert fence.is_valid('{"answer": "ok"}')
    assert not fence.is_valid("{}")


def test_constructor_rejects_contract_and_individual_options() -> None:
    with pytest.raises(ValueError, match="not both"):
        LangFence(
            contract=OutputContract(format=ChoiceConstraint(["a"])),
            language=LanguagePolicy(include=["en"]),
        )


def test_client_returns_langfence_client_wired_with_contract() -> None:
    fence = LangFence(
        format=JsonSchemaConstraint(schema=_SCHEMA),
        language=LanguagePolicy(include=["en"], min_confidence=0.2),
    )

    client = fence.client(base_url="https://provider.test", model="model-a", provider="vllm")

    try:
        assert isinstance(client, LangFenceClient)
        assert client.contract is fence.contract
        assert client.profile == "vllm"
        assert client.model == "model-a"
        assert client.base_url == "https://provider.test"
    finally:
        client.close()


def test_chat_happy_path_end_to_end_with_injected_client() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return _openai_response('{"answer": "ok"}')

    fence = LangFence(format=JsonSchemaConstraint(schema=_SCHEMA))
    injected = httpx.Client(transport=httpx.MockTransport(handler))

    result = fence.chat(
        [{"role": "user", "content": "prompt"}],
        base_url="https://provider.test",
        model="model-a",
        provider="vllm",
        client=injected,
    )

    assert result.ok
    assert result.parsed == {"answer": "ok"}
    assert len(requests) == 1


def test_chat_does_not_close_injected_httpx_client() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return _openai_response("approved")

    fence = LangFence(format=ChoiceConstraint(["approved"]))
    injected = httpx.Client(transport=httpx.MockTransport(handler))

    fence.chat(
        [{"role": "user", "content": "prompt"}],
        base_url="https://provider.test/v1",
        model="model-a",
        provider="openai-compatible",
        client=injected,
    )

    # fence.chat closes the LangFenceClient it created, but that client does not
    # own the injected httpx client, so the caller's client stays open.
    assert not injected.is_closed
    reused = injected.post("https://provider.test/v1/chat/completions", json={})
    assert reused.status_code == 200
    assert call_count == 2


def test_compile_produces_provider_payload() -> None:
    fence = LangFence(format=ChoiceConstraint(["approved", "rejected"]))

    compiled = fence.compile("sglang", [{"role": "user", "content": "classify"}])

    assert compiled.payload["extra_body"]["regex"] == "(?:approved|rejected)"
