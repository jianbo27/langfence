import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from langfence import cli

runner = CliRunner()


class MockResponse:
    def __init__(self, data: dict[str, Any], status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code
        self.text = json.dumps(data)

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._data


class MockClient:
    responses: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def __enter__(self) -> "MockClient":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def post(
        self,
        endpoint: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> MockResponse:
        self.requests.append({"endpoint": endpoint, "json": json, "headers": headers})
        return MockResponse(self.responses.pop(0))


def test_chat_openai_defaults_to_validation_summary_without_text(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        """
format:
  type: regex
  pattern: "^ok$"
"""
    )
    MockClient.responses = [
        {"choices": [{"message": {"content": "ok"}}]},
    ]
    MockClient.requests = []
    monkeypatch.setattr(cli.httpx, "Client", MockClient)

    result = runner.invoke(
        cli.app,
        [
            "chat",
            "--provider",
            "openai",
            "--base-url",
            "https://example.test/v1",
            "--model",
            "gpt-test",
            "--contract",
            str(contract),
            "--prompt",
            "say ok",
        ],
    )

    assert result.exit_code == 0
    output = json.loads(result.output)
    assert output["validation"]["ok"] is True
    assert output["redacted"] is True
    assert "text" not in output
    assert MockClient.requests[0]["endpoint"] == "https://example.test/v1/chat/completions"
    assert MockClient.requests[0]["json"]["model"] == "gpt-test"
    assert MockClient.requests[0]["json"]["messages"][-1] == {
        "role": "user",
        "content": "say ok",
    }


def test_chat_show_sensitive_prints_text(tmp_path: Path, monkeypatch: Any) -> None:
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        """
format:
  type: choice
  choices: ["yes", "no"]
"""
    )
    MockClient.responses = [
        {"choices": [{"message": {"content": "yes"}}]},
    ]
    MockClient.requests = []
    monkeypatch.setattr(cli.httpx, "Client", MockClient)

    result = runner.invoke(
        cli.app,
        [
            "chat",
            "--provider",
            "litellm",
            "--base-url",
            "http://localhost:4000",
            "--model",
            "router-model",
            "--contract",
            str(contract),
            "--prompt",
            "answer",
            "--show-sensitive",
        ],
    )

    assert result.exit_code == 0
    output = json.loads(result.output)
    assert output["redacted"] is False
    assert output["text"] == "yes"


def test_chat_retries_until_output_valid(tmp_path: Path, monkeypatch: Any) -> None:
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        """
format:
  type: regex
  pattern: "^ok$"
"""
    )
    MockClient.responses = [
        {"choices": [{"message": {"content": "bad"}}]},
        {"choices": [{"message": {"content": "ok"}}]},
    ]
    MockClient.requests = []
    monkeypatch.setattr(cli.httpx, "Client", MockClient)

    result = runner.invoke(
        cli.app,
        [
            "chat",
            "--provider",
            "vllm",
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "local-model",
            "--contract",
            str(contract),
            "--prompt",
            "answer",
            "--max-retries",
            "1",
        ],
    )

    assert result.exit_code == 0
    output = json.loads(result.output)
    assert output["attempts"] == 2
    assert output["validation"]["ok"] is True
    assert len(MockClient.requests) == 2
    retry_messages = MockClient.requests[1]["json"]["messages"]
    repair_messages = [
        message["content"]
        for message in retry_messages
        if message["role"] == "system"
        and "Previous response failed output contract validation" in message["content"]
    ]
    assert len(repair_messages) == 1
    assert "bad" not in json.dumps(retry_messages)


def test_chat_anthropic_uses_messages_endpoint_and_headers(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        """
prompt_instruction: "Return only the requested token."
format:
  type: regex
  pattern: "^ok$"
"""
    )
    messages_file = tmp_path / "messages.json"
    messages_file.write_text(
        json.dumps(
            [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "say ok"},
            ]
        )
    )
    MockClient.responses = [
        {"content": [{"type": "text", "text": "ok"}]},
    ]
    MockClient.requests = []
    monkeypatch.setattr(cli.httpx, "Client", MockClient)
    monkeypatch.setenv("ANTHROPIC_TEST_KEY", "secret-key")

    result = runner.invoke(
        cli.app,
        [
            "chat",
            "--provider",
            "anthropic",
            "--base-url",
            "https://api.anthropic.test/v1",
            "--model",
            "claude-test",
            "--contract",
            str(contract),
            "--messages",
            str(messages_file),
            "--api-key-env",
            "ANTHROPIC_TEST_KEY",
        ],
    )

    assert result.exit_code == 0
    request = MockClient.requests[0]
    assert request["endpoint"] == "https://api.anthropic.test/v1/messages"
    assert request["headers"]["x-api-key"] == "secret-key"
    assert request["headers"]["anthropic-version"] == cli.ANTHROPIC_VERSION
    assert request["json"]["model"] == "claude-test"
    assert request["json"]["messages"] == [{"role": "user", "content": "say ok"}]
    assert "Be terse." in request["json"]["system"]
    assert "Return only the requested token." in request["json"]["system"]


def test_chat_openai_uses_bearer_api_key_header(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        """
format:
  type: regex
  pattern: "^ok$"
"""
    )
    MockClient.responses = [
        {"choices": [{"message": {"content": "ok"}}]},
    ]
    MockClient.requests = []
    monkeypatch.setattr(cli.httpx, "Client", MockClient)
    monkeypatch.setenv("OPENAI_TEST_KEY", "secret-key")

    result = runner.invoke(
        cli.app,
        [
            "chat",
            "--provider",
            "openai",
            "--base-url",
            "https://api.openai.test/v1",
            "--model",
            "gpt-test",
            "--contract",
            str(contract),
            "--prompt",
            "say ok",
            "--api-key-env",
            "OPENAI_TEST_KEY",
        ],
    )

    assert result.exit_code == 0
    assert MockClient.requests[0]["headers"]["Authorization"] == "Bearer secret-key"
