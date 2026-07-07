from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from langfence.clients import LangFenceClient
from langfence.serialization import load_contract

_JSON_LANGUAGE_CONTRACT = """
format:
  type: json_schema
  name: localized_answer
  schema:
    type: object
    properties:
      answer:
        type: string
    required: ["answer"]
    additionalProperties: false
language:
  include: ["zh"]
  action: repair
  min_confidence: 0.2
"""

_CHOICE_CONTRACT = """
format:
  type: choice
  choices: ["approved", "rejected"]
"""


def _openai_response(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_yaml_contract_drives_repair_retry_to_valid_chinese_json(tmp_path: Path) -> None:
    contract_file = tmp_path / "contract.yaml"
    contract_file.write_text(_JSON_LANGUAGE_CONTRACT, encoding="utf-8")
    contract = load_contract(contract_file)

    requests: list[dict[str, Any]] = []
    responses = iter(
        [
            _openai_response("not valid json at all"),
            _openai_response('{"answer": "这是一个中文回答。"}'),
        ]
    )

    def engine(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return next(responses)

    client = LangFenceClient(
        provider="vllm",
        base_url="https://engine.test",
        model="model-a",
        contract=contract,
        max_retries=1,
        client=httpx.Client(transport=httpx.MockTransport(engine)),
    )

    result = client.chat([{"role": "user", "content": "请回答"}])

    assert result.ok
    assert result.attempts == 2
    assert result.parsed == {"answer": "这是一个中文回答。"}
    assert len(requests) == 2

    # The second request must carry a repair system message that the first did not.
    first_repair = [
        message
        for message in requests[0]["messages"]
        if message["role"] == "system"
        and "Previous response failed output contract validation" in message["content"]
    ]
    second_repair = [
        message
        for message in requests[1]["messages"]
        if message["role"] == "system"
        and "Previous response failed output contract validation" in message["content"]
    ]
    assert first_repair == []
    assert len(second_repair) == 1


def test_yaml_choice_contract_compiles_unanchored_sglang_regex(tmp_path: Path) -> None:
    contract_file = tmp_path / "contract.yaml"
    contract_file.write_text(_CHOICE_CONTRACT, encoding="utf-8")
    contract = load_contract(contract_file)

    requests: list[dict[str, Any]] = []

    def engine(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return _openai_response("approved")

    client = LangFenceClient(
        provider="sglang",
        base_url="https://engine.test",
        model="model-a",
        contract=contract,
        client=httpx.Client(transport=httpx.MockTransport(engine)),
    )

    result = client.chat([{"role": "user", "content": "classify"}])

    assert result.ok
    assert result.text == "approved"
    # SGLang choices compile to an unanchored alternation flattened onto the body.
    assert requests[0]["regex"] == "(?:approved|rejected)"
    assert "extra_body" not in requests[0]
