# Architecture

LangFence has three layers:

1. Contract model: provider-neutral dataclasses for output format constraints and language policy.
2. Provider adapters: compilation to vLLM, SGLang, OpenAI-compatible, LiteLLM, or
   Anthropic-compatible request payloads.
3. Runtime enforcement: HTTP client, post-generation validation, retry, CLI, and proxy.

Human-facing diagnostics redact prompt text and common secret fields by default.

## Provider Differences

OpenAI-compatible serving fields are not fully portable across providers.

| Capability | vLLM | SGLang | OpenAI-compatible / LiteLLM | Anthropic-compatible |
| --- | --- | --- | --- | --- |
| JSON schema | `response_format` (type json_schema) | `response_format.type=json_schema` | standard `response_format.type=json_schema` | system instruction + validation |
| Regex | `extra_body.structured_outputs.regex` | `extra_body.regex` | system instruction + validation | system instruction + validation |
| Choice | `extra_body.structured_outputs.choice` | compiled to regex | system instruction + validation | system instruction + validation |
| Grammar | `extra_body.structured_outputs.grammar` | `extra_body.ebnf` | system instruction + validation | system instruction + validation |
| Structural tag | `extra_body.structured_outputs.structural_tag` (JSON-encoded string) | `response_format.type=structural_tag` | system instruction + validation | system instruction + validation |

Native/offline fields also differ. vLLM uses `SamplingParams(structured_outputs=...)`;
SGLang uses sampling params such as `json_schema`, `regex`, `ebnf`, and `structural_tag`.

## Language Policy Boundary

Natural-language policies are not hard constrained-decoding guarantees. The library can:

- inject a system instruction,
- add formal constraints where possible,
- detect language after generation,
- return policy issues and retry/repair decisions.

It cannot prove that a long answer is semantically "only Chinese" or "not English".

## Runtime Flow

```text
YAML/Python contract
  -> OutputContract
  -> provider adapter
  -> provider request payload
  -> vLLM/SGLang/OpenAI-compatible/LiteLLM/Anthropic-compatible endpoint
  -> validate_output()
  -> ValidationResult
  -> accept/fail/retry
```

## Extension Points

- Add a provider by implementing an adapter under `src/langfence/adapters/`.
- Add runtime behavior in `src/langfence/clients/http.py`.
- Add a language detector by extending `detect_language()`.
- Add a validator by extending `validate_output()` for a new constraint kind.
- Add service behavior in `src/langfence/service/app.py`.
