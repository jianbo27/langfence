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
from langfence import LangFenceClient, RegexConstraint, OutputContract
import os

contract = OutputContract(format=RegexConstraint(r"^(approved|rejected)$"))

client = LangFenceClient(
    provider="openai-compatible",
    base_url="http://localhost:8000/v1",
    model="local-model",
    contract=contract,
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
client = LangFenceClient(
    provider="anthropic",
    base_url="https://api.anthropic.com/v1",
    model="claude-compatible",
    contract=contract,
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
langfence proxy --provider vllm --base-url http://localhost:8000/v1
```

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

## Development

```bash
pytest
ruff check .
mypy src/langfence
```

See `docs/architecture.md` for design notes and `docs/privacy.md` for privacy rules.
