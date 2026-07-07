from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from langfence import GrammarConstraint, OutputContract, RegexConstraint
from langfence.privacy import REDACTED
from langfence.service.app import create_app

_OK_CONTRACT = {"format": {"kind": "regex", "pattern": "^ok$"}}


def _mock_async_client(handler: Callable[..., httpx.Response]) -> type:
    class MockAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> MockAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(self, *args: Any, **kwargs: Any) -> httpx.Response:
            return handler(*args, **kwargs)

    return MockAsyncClient


def _patch_provider(monkeypatch: Any, handler: Callable[..., httpx.Response]) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", _mock_async_client(handler))


def test_healthz_reports_ok() -> None:
    with TestClient(create_app(provider="vllm", base_url="https://provider.test/v1")) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_app_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported provider: bogus"):
        create_app(provider="bogus", base_url="https://provider.test/v1")


def test_proxy_rejects_non_object_json_body() -> None:
    with TestClient(
        create_app(
            provider="vllm",
            base_url="https://provider.test/v1",
            default_contract=OutputContract(format=RegexConstraint(r"ok")),
        )
    ) as client:
        response = client.post("/v1/chat/completions", json=["not", "an", "object"])

    assert response.status_code == 400
    assert response.json()["detail"] == "Request body must be a JSON object."


def test_proxy_rejects_malformed_json_body() -> None:
    with TestClient(
        create_app(
            provider="vllm",
            base_url="https://provider.test/v1",
            default_contract=OutputContract(format=RegexConstraint(r"ok")),
        )
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            content="{bad json",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Request body must be valid JSON."


def test_proxy_rejects_streaming_requests() -> None:
    with TestClient(
        create_app(
            provider="vllm",
            base_url="https://provider.test/v1",
            default_contract=OutputContract(format=RegexConstraint(r"ok")),
        )
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Streaming is not supported by the LangFence proxy."


def test_proxy_returns_502_for_provider_non_json_body(monkeypatch: Any) -> None:
    _patch_provider(monkeypatch, lambda *a, **k: httpx.Response(200, text="not json"))
    with TestClient(
        create_app(
            provider="vllm",
            base_url="https://provider.test/v1",
            default_contract=OutputContract(format=RegexConstraint(r"ok")),
        )
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "say ok"}]},
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "Provider returned a non-JSON body"


def test_validate_redacts_parsed_output_by_default() -> None:
    with TestClient(create_app(provider="vllm", base_url="https://provider.test/v1")) as client:
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


def test_validate_rejects_bad_contract_with_400() -> None:
    with TestClient(create_app(provider="vllm", base_url="https://provider.test/v1")) as client:
        response = client.post(
            "/validate",
            json={"contract": {"format": {"kind": "nonsense"}}, "output": "x"},
        )

    assert response.status_code == 400
    assert "nonsense" in response.json()["detail"]


def test_compile_endpoint_auto_selects_anthropic_mode() -> None:
    with TestClient(create_app(provider="vllm", base_url="https://provider.test/v1")) as client:
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


def test_compile_endpoint_rejects_unknown_provider_with_400() -> None:
    with TestClient(create_app(provider="vllm", base_url="https://provider.test/v1")) as client:
        response = client.post(
            "/compile",
            json={
                "provider": "bogus",
                "contract": {"format": {"kind": "regex", "pattern": "^ok$"}},
            },
        )

    assert response.status_code == 400
    assert "bogus" in response.json()["detail"]


def test_proxy_hides_provider_error_body_by_default(monkeypatch: Any) -> None:
    _patch_provider(
        monkeypatch, lambda *a, **k: httpx.Response(500, text="sensitive provider body")
    )
    with TestClient(
        create_app(
            provider="vllm",
            base_url="https://provider.test/v1",
            default_contract=OutputContract(format=RegexConstraint(r"ok")),
        )
    ) as client:
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
    _patch_provider(monkeypatch, lambda *a, **k: httpx.Response(200, json={"choices": []}))
    with TestClient(
        create_app(
            provider="vllm",
            base_url="https://provider.test/v1",
            default_contract=OutputContract(),
        )
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "say ok"}]},
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "Provider response is missing message content."


def test_proxy_handles_list_of_parts_content(monkeypatch: Any) -> None:
    _patch_provider(
        monkeypatch,
        lambda *a, **k: httpx.Response(
            200,
            json={"choices": [{"message": {"content": [{"type": "text", "text": "ok"}]}}]},
        ),
    )
    with TestClient(
        create_app(
            provider="vllm",
            base_url="https://provider.test/v1",
            default_contract=OutputContract(format=RegexConstraint(r"^ok$")),
        )
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "say ok"}]},
        )

    assert response.status_code == 200
    assert response.json()["output_contract"]["ok"] is True


def test_proxy_accepts_provider_enforced_grammar_for_vllm(monkeypatch: Any) -> None:
    _patch_provider(
        monkeypatch,
        lambda *a, **k: httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
    )
    with TestClient(
        create_app(
            provider="vllm",
            base_url="https://provider.test/v1",
            default_contract=OutputContract(format=GrammarConstraint('root ::= "ok"')),
        )
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "say ok"}]},
        )

    assert response.status_code == 200
    assert response.json()["output_contract"]["ok"] is True


def test_proxy_uppercase_provider_validates_grammar_locally(monkeypatch: Any) -> None:
    # Grammar is only provider-enforced for vllm/sglang. With a normalized provider
    # value, an uppercase "VLLM" must still be treated as vllm and skip the
    # unavailable-validation path rather than emitting grammar.validation_unavailable.
    _patch_provider(
        monkeypatch,
        lambda *a, **k: httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
    )
    with TestClient(
        create_app(
            provider="VLLM",
            base_url="https://provider.test/v1",
            default_contract=OutputContract(format=GrammarConstraint('root ::= "ok"')),
        )
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "say ok"}]},
        )

    assert response.status_code == 200
    assert response.json()["output_contract"]["ok"] is True


def test_proxy_returns_200_with_output_contract_field_on_violation(monkeypatch: Any) -> None:
    _patch_provider(
        monkeypatch,
        lambda *a, **k: httpx.Response(200, json={"choices": [{"message": {"content": "nope"}}]}),
    )
    with TestClient(
        create_app(
            provider="vllm",
            base_url="https://provider.test/v1",
            default_contract=OutputContract(format=RegexConstraint(r"^ok$")),
        )
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "say ok"}]},
        )

    assert response.status_code == 200
    contract = response.json()["output_contract"]
    assert contract["ok"] is False
    assert contract["issues"]
    assert contract["issues"][0]["severity"] == "error"


def test_proxy_violation_status_code_returns_configured_status(monkeypatch: Any) -> None:
    _patch_provider(
        monkeypatch,
        lambda *a, **k: httpx.Response(200, json={"choices": [{"message": {"content": "nope"}}]}),
    )
    with TestClient(
        create_app(
            provider="vllm",
            base_url="https://provider.test/v1",
            default_contract=OutputContract(format=RegexConstraint(r"^ok$")),
            violation_status_code=422,
        )
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "say ok"}]},
        )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["ok"] is False
    assert detail["issues"]


def test_proxy_per_request_contract_override(monkeypatch: Any) -> None:
    _patch_provider(
        monkeypatch,
        lambda *a, **k: httpx.Response(200, json={"choices": [{"message": {"content": "yes"}}]}),
    )
    with TestClient(
        create_app(
            provider="vllm",
            base_url="https://provider.test/v1",
            default_contract=OutputContract(format=RegexConstraint(r"^ok$")),
        )
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "say yes"}],
                "x-output-contract": {"format": {"kind": "choice", "choices": ["yes", "no"]}},
            },
        )

    assert response.status_code == 200
    assert response.json()["output_contract"]["ok"] is True


def test_proxy_rejects_non_dict_contract_override_with_400(monkeypatch: Any) -> None:
    _patch_provider(
        monkeypatch,
        lambda *a, **k: httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
    )
    with TestClient(
        create_app(
            provider="vllm",
            base_url="https://provider.test/v1",
            default_contract=OutputContract(format=RegexConstraint(r"^ok$")),
        )
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "say ok"}],
                "x-output-contract": "not a dict",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "x-output-contract must be a JSON object."
