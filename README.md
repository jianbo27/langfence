# LangFence

LangFence puts a small, explicit contract around LLM output.

It compiles format constraints into the strongest request fields a provider supports,
adds language guidance when useful, and validates the returned text locally. The goal
is not to pretend natural language can be perfectly constrained. The goal is to make
format and language expectations visible, testable, and easy to use with local or
OpenAI-compatible serving stacks.

## What It Supports

- vLLM and SGLang, out of the box
- LiteLLM and generic OpenAI-compatible `/chat/completions` endpoints
- Anthropic-compatible `/messages` endpoints
- Python API, CLI, and an optional OpenAI-compatible sidecar proxy
- JSON Schema, regex, choice, grammar/EBNF, and structural tag contracts
- Language include/exclude policies, checked after generation
- Redacted CLI/service diagnostics by default

## Install

```bash
pip install langfence
```

For local development:

```bash
pip install -e ".[dev,all]"
```

## Python API

For most Python code, start with the `LangFence` facade:

```python
from langfence import LangFence, LanguagePolicy

fence = LangFence(language=LanguagePolicy(include=["zh"], exclude=["en"], min_confidence=0.2))

result = fence.validate("这是一个中文回答。")
assert result.valid
```

Use `compile_request` when you only want the provider payload:

```python
from langfence import JsonSchemaConstraint, LanguagePolicy, OutputContract, compile_request

contract = OutputContract(
    format=JsonSchemaConstraint(
        name="answer",
        schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}, "language": {"enum": ["zh"]}},
            "required": ["answer", "language"],
            "additionalProperties": False,
        },
    ),
    language=LanguagePolicy(include=["zh"], exclude=["en"], min_confidence=0.2),
)

compiled = compile_request(
    "vllm",
    [{"role": "user", "content": "用中文解释 constrained decoding。"}],
    contract,
    base_payload={"model": "Qwen/Qwen2.5-7B-Instruct"},
)
```

Use `LangFenceClient` when you want request, retry, extraction, and validation in one
place:

```python
from langfence import LangFence, RegexConstraint

fence = LangFence(format=RegexConstraint(r"^(approved|rejected)$"))

client = fence.client(
    provider="openai-compatible",
    base_url="http://localhost:8000/v1",
    model="local-model",
    max_retries=1,
)

result = client.chat([{"role": "user", "content": "Return approved or rejected."}])
assert result.ok
print(result.text)
```

Provider names accepted by the client:

- `vllm`
- `sglang`
- `openai` or `openai-compatible`
- `litellm`
- `anthropic` or `anthropic-compatible`

For Anthropic-compatible endpoints, use the Messages API base URL:

```python
import os

client = fence.client(
    provider="anthropic",
    base_url="https://api.anthropic.com/v1",
    model="claude-compatible",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)
```

## CLI

Compile a request payload:

```bash
langfence compile --provider vllm --contract examples/contract.zh.yaml
langfence compile --provider sglang --contract examples/contract.zh.yaml
```

Call a model and validate its output:

```bash
langfence chat \
  --provider vllm \
  --base-url http://localhost:8000/v1 \
  --model local-model \
  --contract examples/contract.zh.yaml \
  --prompt "用中文回答。"
```

Use `--api-key-env OPENAI_API_KEY` or `--api-key-env ANTHROPIC_API_KEY` to read keys
from the environment. CLI output redacts model text by default; add
`--show-sensitive` only in a trusted local shell.

Validate saved output:

```bash
langfence validate --contract examples/contract.zh.yaml --input output.txt
```

Run the optional OpenAI-compatible proxy:

```bash
pip install "langfence[service]"
langfence proxy --provider vllm --base-url http://localhost:8000/v1
```

## Proxy Service

The optional proxy is a thin OpenAI-compatible sidecar. It compiles the contract into
the upstream request, forwards to the configured provider, and validates the returned
text before responding.

- `POST /v1/chat/completions` proxies a chat request. Pass a per-request contract as an
  `x-output-contract` object in the request body, or start the proxy with `--contract`
  for a default. By default a contract violation still returns HTTP 200 with the parsed
  provider response plus an added `output_contract` field (`{"ok": ..., "issues": [...]}`),
  so vanilla OpenAI SDK callers keep working. Pass `--violation-status 422` (any status)
  to instead fail the request with that status and a body carrying the validation issues.
- `POST /compile` returns the compiled provider payload for a contract.
- `POST /validate` validates saved output text against a contract.
- `GET /healthz` returns `{"status": "ok"}`.

Streaming is not supported: a request with `stream: true` is rejected with HTTP 400,
because validation needs the complete output. Raw provider error bodies are hidden by
default; pass `--include-provider-error-body` to surface them in a trusted environment.

## Provider Behavior

| Provider | Format handling | Language handling |
| --- | --- | --- |
| vLLM | Uses vLLM structured output fields where available | Prompt guidance + local validation |
| SGLang | Uses SGLang regex/JSON/EBNF/structural fields where available | Prompt guidance + local validation |
| OpenAI-compatible | Uses standard JSON Schema `response_format`; other formats are validated after generation | Prompt guidance + local validation |
| LiteLLM | Same portable OpenAI-compatible behavior | Prompt guidance + local validation |
| Anthropic-compatible | Uses Messages `system` guidance; validation runs after generation | Prompt guidance + local validation |

LangFence does not log prompts, outputs, or provider responses. HTTP error bodies are
hidden by default because providers may echo request content.

LangFence validates text returned to the caller. It cannot inspect or constrain hidden
reasoning traces or internal chain-of-thought that a provider does not expose. If a
local reasoning model returns a leading visible block such as `<think>...</think>`,
LangFence strips that block before validation by default and checks the final answer.
This keeps reasoning-model output quality intact: no extra constrained decoding or
prompt pressure is added just to control the language of private reasoning.

## Language Policy

`LanguagePolicy` chooses a language detector with the `detector` field:

- `detector="heuristic"` (default) uses the built-in, dependency-free detector.
- `detector="lingua"` uses `lingua-language-detector` and raises at construction if that
  optional dependency is not installed. Install it with `pip install "langfence[language]"`.
- `detector="auto"` uses lingua when it is available and falls back to the heuristic
  detector otherwise.

Two thresholds are checked independently:

- `min_confidence` (default 0.75) is the minimum confidence for the detected primary
  language of the output.
- `exclude_threshold` (default 0.20) is the minimum confidence at which an excluded
  language is treated as present.

## Client Behavior

`LangFenceClient` handles retries, extraction, and validation:

- `max_tokens` is only sent when you set it explicitly. The Anthropic transport, which
  requires the field, falls back to 1024.
- `max_retries` is a shared budget covering both validation-driven retries and retryable
  transport failures (HTTP 408/429/502/503/504 and network errors), using exponential
  backoff between attempts.
- Streaming is not supported: passing `stream=True` raises `ValueError`, because
  validation needs the complete output.

## Development

```bash
pytest
ruff check .
mypy src/langfence
```

See `docs/architecture.md` for design notes and `docs/privacy.md` for privacy rules.
