from __future__ import annotations

from typing import Any

import httpx
from fastapi.testclient import TestClient

from langfence import OutputContract, RegexConstraint
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
