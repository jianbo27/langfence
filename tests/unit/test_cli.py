from __future__ import annotations

import builtins
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import httpx
from typer.testing import CliRunner

from langfence import cli

runner = CliRunner()


def _install_mock_transport(
    monkeypatch: Any,
    handler: Callable[[httpx.Request], httpx.Response],
    requests: list[dict[str, Any]] | None = None,
) -> None:
    """Route the CLI's internal httpx.Client through a MockTransport.

    Mirrors tests/unit/test_client.py: a real httpx.Client with a
    MockTransport exercises the true request/response and HTTP-error paths.
    """

    real_client_cls = httpx.Client

    def wrapped(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(
                {
                    "url": str(request.url),
                    "headers": dict(request.headers),
                    "json": json.loads(request.content) if request.content else None,
                }
            )
        return handler(request)

    def client_factory(*args: Any, **kwargs: Any) -> httpx.Client:
        return real_client_cls(transport=httpx.MockTransport(wrapped))

    monkeypatch.setattr(cli.httpx, "Client", client_factory)


def _openai_response(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def _anthropic_response(content: str) -> httpx.Response:
    return httpx.Response(200, json={"content": [{"type": "text", "text": content}]})


def _write_contract(tmp_path: Path, body: str) -> Path:
    contract = tmp_path / "contract.yaml"
    contract.write_text(body)
    return contract


def test_compile_anthropic_compatible_defaults_to_messages_mode(tmp_path: Path) -> None:
    contract = _write_contract(
        tmp_path,
        """
format:
  type: regex
  pattern: "^ok$"
""",
    )

    result = runner.invoke(
        cli.app,
        ["compile", "--provider", "anthropic-compatible", "--contract", str(contract)],
    )

    assert result.exit_code == 0
    output = json.loads(result.output)
    assert output["mode"] == "anthropic"
    assert "messages" in output["payload"]


def test_compile_rejects_unknown_provider_cleanly(tmp_path: Path) -> None:
    contract = _write_contract(tmp_path, "format:\n  type: regex\n  pattern: \"^ok$\"\n")

    result = runner.invoke(
        cli.app,
        ["compile", "--provider", "bogus", "--contract", str(contract)],
    )

    assert result.exit_code != 0
    assert "bogus" in result.output
    assert "Traceback" not in result.output


def test_compile_missing_contract_file_is_clean_error(tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app,
        ["compile", "--provider", "vllm", "--contract", str(tmp_path / "missing.yaml")],
    )

    assert result.exit_code != 0
    assert "not found" in result.output
    assert "Traceback" not in result.output


def test_compile_non_dict_base_payload_is_clean_error(tmp_path: Path) -> None:
    contract = _write_contract(tmp_path, "format:\n  type: regex\n  pattern: \"^ok$\"\n")
    base = tmp_path / "base.json"
    base.write_text("[1, 2, 3]")

    result = runner.invoke(
        cli.app,
        [
            "compile",
            "--provider",
            "vllm",
            "--contract",
            str(contract),
            "--base-payload",
            str(base),
        ],
    )

    assert result.exit_code != 0
    assert "--base-payload must contain a JSON object" in result.output


def test_validate_exit_code_1_and_redacts_parsed_by_default(tmp_path: Path) -> None:
    contract = _write_contract(
        tmp_path,
        """
format:
  type: json_schema
  schema:
    type: object
    properties:
      answer:
        type: string
    required: [answer]
""",
    )
    output_file = tmp_path / "output.txt"
    output_file.write_text('{"answer": 5}')

    result = runner.invoke(
        cli.app,
        ["validate", "--contract", str(contract), "--input", str(output_file)],
    )

    assert result.exit_code == 1
    output = json.loads(result.output)
    assert output["ok"] is False
    assert output["redacted"] is True
    # A falsy-but-valid parse must still redact to the marker, never null.
    assert output["parsed"] == cli.REDACTED


def test_validate_valid_output_exits_zero(tmp_path: Path) -> None:
    contract = _write_contract(tmp_path, "format:\n  type: regex\n  pattern: \"^ok$\"\n")
    output_file = tmp_path / "output.txt"
    output_file.write_text("ok")

    result = runner.invoke(
        cli.app,
        ["validate", "--contract", str(contract), "--input", str(output_file)],
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["ok"] is True


def test_validate_reads_input_from_stdin(tmp_path: Path) -> None:
    contract = _write_contract(tmp_path, "format:\n  type: regex\n  pattern: \"^ok$\"\n")

    result = runner.invoke(
        cli.app,
        ["validate", "--contract", str(contract), "--input", "-"],
        input="ok",
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["ok"] is True


def test_chat_openai_defaults_to_validation_summary_without_text(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    contract = _write_contract(tmp_path, "format:\n  type: regex\n  pattern: \"^ok$\"\n")
    requests: list[dict[str, Any]] = []
    _install_mock_transport(monkeypatch, lambda request: _openai_response("ok"), requests)

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
    assert requests[0]["url"] == "https://example.test/v1/chat/completions"
    assert requests[0]["json"]["model"] == "gpt-test"
    assert requests[0]["json"]["messages"][-1] == {"role": "user", "content": "say ok"}


def test_chat_show_sensitive_prints_text(tmp_path: Path, monkeypatch: Any) -> None:
    contract = _write_contract(
        tmp_path,
        "format:\n  type: choice\n  choices: [\"yes\", \"no\"]\n",
    )
    _install_mock_transport(monkeypatch, lambda request: _openai_response("yes"))

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
    contract = _write_contract(tmp_path, "format:\n  type: regex\n  pattern: \"^ok$\"\n")
    requests: list[dict[str, Any]] = []
    responses = iter([_openai_response("bad"), _openai_response("ok")])
    _install_mock_transport(monkeypatch, lambda request: next(responses), requests)

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
    assert len(requests) == 2
    retry_messages = requests[1]["json"]["messages"]
    repair_messages = [
        message["content"]
        for message in retry_messages
        if message["role"] == "system"
        and "Previous response failed output contract validation" in message["content"]
    ]
    assert len(repair_messages) == 1
    assert "bad" not in json.dumps(retry_messages)


def test_chat_invalid_output_exits_with_code_1(tmp_path: Path, monkeypatch: Any) -> None:
    contract = _write_contract(tmp_path, "format:\n  type: regex\n  pattern: \"^ok$\"\n")
    _install_mock_transport(monkeypatch, lambda request: _openai_response("nope"))

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
        ],
    )

    assert result.exit_code == 1
    output = json.loads(result.output)
    assert output["validation"]["ok"] is False


def test_chat_provider_http_error_is_clean_error(tmp_path: Path, monkeypatch: Any) -> None:
    contract = _write_contract(tmp_path, "format:\n  type: regex\n  pattern: \"^ok$\"\n")
    _install_mock_transport(
        monkeypatch,
        lambda request: httpx.Response(500, text="sensitive provider body"),
    )

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
        ],
    )

    assert result.exit_code != 0
    # A clean ClickException, not a raw traceback, and no leaked provider body.
    assert isinstance(result.exception, click.ClickException)
    message = result.exception.format_message()
    assert "HTTP 500" in message
    assert "sensitive provider body" not in message


def test_chat_reads_messages_from_stdin(tmp_path: Path, monkeypatch: Any) -> None:
    contract = _write_contract(tmp_path, "format:\n  type: regex\n  pattern: \"^ok$\"\n")
    requests: list[dict[str, Any]] = []
    _install_mock_transport(monkeypatch, lambda request: _openai_response("ok"), requests)

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
            "--messages",
            "-",
        ],
        input=json.dumps([{"role": "user", "content": "say ok"}]),
    )

    assert result.exit_code == 0
    assert requests[0]["json"]["messages"][-1] == {"role": "user", "content": "say ok"}


def test_chat_anthropic_uses_messages_endpoint_and_headers(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    contract = _write_contract(
        tmp_path,
        """
prompt_instruction: "Return only the requested token."
format:
  type: regex
  pattern: "^ok$"
""",
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
    requests: list[dict[str, Any]] = []
    _install_mock_transport(monkeypatch, lambda request: _anthropic_response("ok"), requests)
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
    request = requests[0]
    assert request["url"] == "https://api.anthropic.test/v1/messages"
    assert request["headers"]["x-api-key"] == "secret-key"
    assert request["headers"]["anthropic-version"] == cli.ANTHROPIC_VERSION
    assert request["json"]["model"] == "claude-test"
    assert request["json"]["messages"] == [{"role": "user", "content": "say ok"}]
    assert "Be terse." in request["json"]["system"]
    assert "Return only the requested token." in request["json"]["system"]


def test_chat_openai_uses_bearer_api_key_header(tmp_path: Path, monkeypatch: Any) -> None:
    contract = _write_contract(tmp_path, "format:\n  type: regex\n  pattern: \"^ok$\"\n")
    requests: list[dict[str, Any]] = []
    _install_mock_transport(monkeypatch, lambda request: _openai_response("ok"), requests)
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
    assert requests[0]["headers"]["authorization"] == "Bearer secret-key"


def test_proxy_requires_service_extra(tmp_path: Path, monkeypatch: Any) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "uvicorn" or name.startswith("langfence.service"):
            raise ImportError("No module named 'uvicorn'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = runner.invoke(
        cli.app,
        [
            "proxy",
            "--provider",
            "vllm",
            "--base-url",
            "http://localhost:8000/v1",
        ],
    )

    assert result.exit_code != 0
    assert "service" in result.output
    assert "Traceback" not in result.output


def test_proxy_passes_violation_status_to_create_app(tmp_path: Path, monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_create_app(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "app-sentinel"

    def fake_run(app: Any, **kwargs: Any) -> None:
        captured["run_app"] = app

    import sys
    import types

    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = fake_run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setattr("langfence.service.app.create_app", fake_create_app)

    result = runner.invoke(
        cli.app,
        [
            "proxy",
            "--provider",
            "vllm",
            "--base-url",
            "http://localhost:8000/v1",
            "--violation-status",
            "422",
        ],
    )

    assert result.exit_code == 0
    assert captured["provider"] == "vllm"
    assert captured["violation_status_code"] == 422
    assert captured["run_app"] == "app-sentinel"
