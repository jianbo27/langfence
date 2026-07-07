from __future__ import annotations

from typing import Any

import httpx
from fastapi.testclient import TestClient

from langfence import GrammarConstraint, OutputContract, RegexConstraint
from langfence.privacy import REDACTED
from langfence.service.app import create_app


def test_proxy_rejects_non_object_json_body() -> None:
    client = TestClient(
        create_app(
            provider="vllm",
            base_url="https://provider.test/v1",
            default_contract=OutputContract(format=RegexConstraint(r"ok")),
        )
    )

    response = client.post("/v1/chat/completions", json=["not", "an", "object"])

    assert response.status_code == 400
    assert response.json()["detail"] == "Request body must be a JSON object."


def test_proxy_rejects_malformed_json_body() -> None:
    client = TestClient(
        create_app(
            provider="vllm",
            base_url="https://provider.test/v1",
            default_contract=OutputContract(format=RegexConstraint(r"ok")),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        content="{bad json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Request body must be valid JSON."


def test_proxy_returns_502_for_provider_non_json_body(monkeypatch: Any) -> None:
    class MockAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> MockAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(self, *args: Any, **kwargs: Any) -> httpx.Response:
            return httpx.Response(200, text="not json")

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)
    client = TestClient(
        create_app(
            provider="vllm",
            base_url="https://provider.test/v1",
            default_contract=OutputContract(format=RegexConstraint(r"ok")),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "say ok"}]},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "Provider returned a non-JSON body"


def test_validate_redacts_parsed_output_by_default() -> None:
    client = TestClient(create_app(provider="vllm", base_url="https://provider.test/v1"))

    response = client.post(
        "/validate",
        json={
            "contract": {
                "format": {
                    "kind": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                }
            },
            "output": '{"answer":"private answer"}',
        },
    )

    assert response.status_code == 200
    assert response.json()["parsed"] == REDACTED


def test_compile_endpoint_auto_selects_anthropic_mode() -> None:
    client = TestClient(create_app(provider="vllm", base_url="https://provider.test/v1"))

    response = client.post(
        "/compile",
        json={
            "provider": "anthropic-compatible",
            "messages": [{"role": "user", "content": "say ok"}],
            "contract": {"format": {"kind": "regex", "pattern": "^ok$"}},
            "base_payload": {"model": "claude-compatible"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "anthropic"
    assert body["payload"]["model"] == "claude-compatible"
    assert body["payload"]["messages"][0]["content"] == REDACTED


def test_proxy_hides_provider_error_body_by_default(monkeypatch: Any) -> None:
    class MockAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> MockAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(self, *args: Any, **kwargs: Any) -> httpx.Response:
            return httpx.Response(500, text="sensitive provider body")

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)
    client = TestClient(
        create_app(
            provider="vllm",
            base_url="https://provider.test/v1",
            default_contract=OutputContract(format=RegexConstraint(r"ok")),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "say ok"}]},
    )

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["provider_status_code"] == 500
    assert "provider_error_body" not in detail
    assert "sensitive provider body" not in response.text


def test_proxy_returns_502_for_missing_provider_message_content(monkeypatch: Any) -> None:
    class MockAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> MockAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(self, *args: Any, **kwargs: Any) -> httpx.Response:
            return httpx.Response(200, json={"choices": []})

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)
    client = TestClient(
        create_app(
            provider="vllm",
            base_url="https://provider.test/v1",
            default_contract=OutputContract(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "say ok"}]},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "Provider response is missing message content."


def test_proxy_accepts_provider_enforced_grammar_for_vllm(monkeypatch: Any) -> None:
    class MockAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> MockAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(self, *args: Any, **kwargs: Any) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)
    client = TestClient(
        create_app(
            provider="vllm",
            base_url="https://provider.test/v1",
            default_contract=OutputContract(format=GrammarConstraint('root ::= "ok"')),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "say ok"}]},
    )

    assert response.status_code == 200
    assert response.json()["output_contract"]["ok"] is True
