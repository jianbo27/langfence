import json
from typing import Any

import httpx
import pytest

from langfence import ChoiceConstraint, JsonSchemaConstraint, LanguagePolicy, OutputContract
from langfence.clients import LangFenceClient, LangFenceHTTPError


def _openai_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}]},
    )


def _anthropic_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"content": [{"type": "text", "text": content}]},
    )


def test_vllm_profile_uses_adapter_and_chat_completions() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        assert request.url.path == "/chat/completions"
        return _openai_response('{"answer":"ok"}')

    client = LangFenceClient(
        provider="vllm",
        base_url="https://provider.test",
        model="model-a",
        contract=OutputContract(
            format=JsonSchemaConstraint(
                name="answer",
                schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            )
        ),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = client.chat([{"role": "user", "content": "prompt"}], temperature=0)

    assert result.ok
    assert result.parsed == {"answer": "ok"}
    assert requests[0]["model"] == "model-a"
    assert requests[0]["temperature"] == 0
    assert requests[0]["response_format"]["type"] == "json_schema"


def test_litellm_profile_uses_standard_json_schema_only() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return _openai_response('{"answer":"ok"}')

    client = LangFenceClient(
        provider="litellm",
        base_url="https://litellm.test/v1",
        model="model-a",
        contract=OutputContract(
            format=JsonSchemaConstraint(
                name="answer",
                schema={"type": "object", "properties": {"answer": {"type": "string"}}},
            )
        ),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = client.chat([{"role": "user", "content": "prompt"}])

    assert result.ok
    assert requests[0]["response_format"]["json_schema"]["name"] == "answer"
    assert "extra_body" not in requests[0]
    assert "sampling_params" not in requests[0]


def test_openai_compatible_choice_uses_post_validation_without_private_fields() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return _openai_response("approved")

    client = LangFenceClient(
        provider="openai-compatible",
        base_url="https://openai-compatible.test/v1",
        model="model-a",
        contract=OutputContract(format=ChoiceConstraint(["approved", "rejected"])),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = client.chat([{"role": "user", "content": "prompt"}])

    assert result.ok
    assert result.text == "approved"
    assert "response_format" not in requests[0]
    assert "extra_body" not in requests[0]


def test_anthropic_profile_uses_messages_transport_and_prompt_guidance() -> None:
    requests: list[dict[str, Any]] = []
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        requests.append(json.loads(request.content))
        assert request.headers["anthropic-version"] == "2023-06-01"
        return _anthropic_response('{"answer":"ok"}')

    client = LangFenceClient(
        provider="anthropic",
        base_url="https://anthropic-compatible.test/v1",
        model="claude-compatible",
        contract=OutputContract(
            format=JsonSchemaConstraint(
                name="answer",
                schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            ),
            language=LanguagePolicy(include=["en"], min_confidence=0.2),
        ),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = client.chat(
        [
            {"role": "system", "content": "existing system"},
            {"role": "user", "content": "prompt"},
        ],
        max_tokens=64,
    )

    assert result.ok
    assert paths == ["/v1/messages"]
    assert requests[0]["model"] == "claude-compatible"
    assert requests[0]["max_tokens"] == 64
    assert requests[0]["messages"] == [{"role": "user", "content": "prompt"}]
    assert "existing system" in requests[0]["system"]
    assert "Return only valid JSON" in requests[0]["system"]
    assert "Use only these natural languages: en." in requests[0]["system"]
    assert "response_format" not in requests[0]


def test_chat_retries_with_repair_instruction_after_validation_failure() -> None:
    requests: list[dict[str, Any]] = []
    responses = iter([_openai_response("maybe"), _openai_response("approved")])

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return next(responses)

    client = LangFenceClient(
        provider="vllm",
        base_url="https://provider.test",
        model="model-a",
        contract=OutputContract(format=ChoiceConstraint(["approved", "rejected"])),
        max_retries=1,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = client.chat([{"role": "user", "content": "prompt"}])

    assert result.ok
    assert result.attempts == 2
    assert len(requests) == 2
    repair_messages = [
        message["content"]
        for message in requests[1]["messages"]
        if message["role"] == "system"
        and "Previous response failed output contract validation" in message["content"]
    ]
    assert len(repair_messages) == 1
    assert len(repair_messages[0]) < 240
    assert "choice.invalid" in repair_messages[0]


def test_provider_error_body_hidden_by_default() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="sensitive provider body")

    client = LangFenceClient(
        provider="openai-compatible",
        base_url="https://provider.test",
        model="model-a",
        contract=OutputContract(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(LangFenceHTTPError) as exc_info:
        client.chat([{"role": "user", "content": "prompt"}])

    assert exc_info.value.status_code == 500
    assert exc_info.value.error_body is None
