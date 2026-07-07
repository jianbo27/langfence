# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-07

### Fixed

- **`LangFenceClient` now merges `extra_body` into the wire request.** Adapters
  emit engine extension fields (`structured_outputs`, `regex`, `ebnf`) under
  `extra_body` for OpenAI-SDK compatibility, but the built-in HTTP client sent
  the payload verbatim, so vLLM/SGLang silently ignored every non-JSON-schema
  constraint. Regex, choice, grammar, and structural-tag constraints now reach
  constrained decoding.
- YAML contracts with a scalar language value (`include: zh`) load as one
  language code instead of being iterated character-by-character.
- Lingua-based detection now fills per-language confidence scores, so
  `exclude` policies catch secondary-language leaks (e.g. English mixed into a
  Chinese answer) instead of only the top-1 language.
- vLLM `structural_tag` is emitted as the JSON-encoded string the engine
  expects, in both OpenAI and native modes.
- Heuristic language detection covers half-width Katakana, Hangul Jamo and
  compatibility blocks, and CJK Compatibility/Extension-B ideographs; digits,
  punctuation, and symbols no longer dilute scores (`"你好！"` now detects as
  `zh` with confidence 1.0).
- Fenced code blocks, inline code, and URLs are stripped before language
  detection, so ASCII identifiers in code samples no longer trip
  exclude-English policies on legitimate CJK answers.
- Proxy service: streaming requests are rejected with 400 up front instead of
  failing as 502 after paying for generation; malformed contracts and unknown
  providers return 400 instead of 500; provider names are normalized once at
  startup (uppercase `VLLM` no longer breaks provider-enforced validation);
  one shared upstream HTTP client per app instead of one per request;
  non-dict `x-output-contract` overrides return 400.
- CLI no longer renders rich tracebacks with local variables, which could leak
  API keys and prompt content; errors surface as clean messages with proper
  exit codes. Redacted output uses an `is not None` check so falsy-but-valid
  parses (`{}`, `""`, `0`, `false`) show the redaction marker instead of `null`.

### Changed

- `LanguagePolicy` validates its fields at construction: unknown `action` or
  `detector` values and out-of-range thresholds raise `ValueError`; language
  codes are normalized to lowercase. `detector="lingua"` raises at
  construction when the `language` extra is not installed — the new
  `detector="auto"` uses lingua when available and falls back to the
  heuristic.
- New `LanguagePolicy.exclude_threshold` field (default 0.20) controls
  exclusion sensitivity, decoupled from `min_confidence`.
- Include-policy violations on low-signal detections (unknown language or
  confidence below `min_confidence`) are warnings instead of errors, so
  numeric or code-only outputs no longer hard-fail and burn retries.
- Local regex validation no longer applies an implicit `DOTALL`, matching
  engine-side constrained-decoding semantics; use an inline `(?s)` flag when
  needed.
- Constraints validate at construction: invalid JSON Schemas, uncompilable
  regex patterns, empty choice lists, and unknown grammar syntaxes raise
  `ValueError` immediately.
- SGLang adapter rejects non-EBNF grammars with `ValueError` instead of
  sending Lark grammars the engine cannot parse; choice constraints compile to
  an unanchored regex alternation.
- `LangFenceClient`: `max_tokens` is only sent when set explicitly (the
  Anthropic transport, which requires it, defaults to 1024); retryable
  transport failures (HTTP 408/429/502/503/504 with `Retry-After` support, and
  network errors) are retried with exponential backoff within the
  `max_retries` budget; validation issues the local validator cannot check
  (`*.validation_unavailable`) short-circuit instead of burning retries;
  `stream=True` raises `ValueError`; grammar/structural-tag contracts on
  vLLM/SGLang add an explicit "provider-enforced" warning.
- JSON Schema validation issues name the violated rule (e.g.
  `violates the 'enum' rule`) with the schema fragment in metadata, still
  without echoing output values.
- Lingua language models load lazily instead of eagerly preloading all ~75
  languages (~1 GB RSS) on first use.
- Packaging: PEP 639 license metadata, Python 3.14 classifier and CI coverage,
  a dedicated CI job for the `language` extra, and the PyPI publish action
  pinned to a commit SHA.

### Added

- `langfence proxy --violation-status` / `create_app(violation_status_code=...)`
  to return a non-200 status when provider output violates the contract
  (default remains 200 with the `output_contract` field).
- CLI stdin support: `langfence validate --input -` and
  `langfence chat --messages -`.
- Test suite grew from 76 to 217 tests, including wire-level client tests, a
  full provider × constraint golden-payload matrix, integration round-trips,
  and lingua-specific coverage.

### Removed

- Unused `RetryPolicy` and `ContractViolation` dataclasses, the unused
  `ProxyValidation` schema, and `.pypirc.example` (publishing uses OIDC
  trusted publishing).

## [0.1.3] - 2026-07-06

Initial public releases (0.1.0 – 0.1.3): core contract model, vLLM/SGLang/
OpenAI-compatible/Anthropic-compatible adapters, post-generation validation,
CLI, and FastAPI proxy.
