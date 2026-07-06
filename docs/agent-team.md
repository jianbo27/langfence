# Ownership

Use these lanes when splitting work. Keep changes small and reviewable.

| Lane | Ownership | Files |
| --- | --- | --- |
| Product | Use cases, scope, naming | `README.md`, `examples/` |
| Providers | vLLM/SGLang/OpenAI-compatible/Anthropic-compatible request compilation | `adapters/` |
| Constraints | Contract types and compilation rules | `constraints.py`, golden tests |
| Language | include/exclude language checks | `language.py`, `retry.py` |
| Validation | JSON Schema, regex, choice, and issue reporting | `validation.py`, tests |
| Runtime | Python client, retries, response extraction | `clients/http.py` |
| CLI/Proxy | Command line and sidecar service | `cli.py`, `service/` |
| QA | CI, tests, benchmarks | `tests/`, `benchmarks/`, `.github/workflows/ci.yml` |

## Order

1. Define the contract spec and YAML format.
2. Implement provider adapters and golden snapshots.
3. Implement validation and language policy.
4. Add the Python client, CLI, and proxy.
5. Add tests, benchmarks, and CI.
6. Expand integration tests against real vLLM/SGLang, LiteLLM, and Anthropic-compatible servers.

## Review Checklist

- Provider payloads match current official vLLM/SGLang fields.
- Generic OpenAI-compatible and LiteLLM payloads do not include vLLM/SGLang private fields.
- Anthropic-compatible calls use Messages-style payloads and headers.
- SGLang choice fallback is clear and tested.
- Language policy failures are reported as post-generation validation issues.
- No documentation claims natural-language enforcement is perfect.
- Prompts, outputs, tokens, and provider error bodies are not logged or printed by default.
- GPU integration tests stay optional in default CI.
