from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import httpx

from langfence.adapters import compile_request
from langfence.constraints import GrammarConstraint, StructuralTagConstraint
from langfence.contracts import OutputContract, RequestMode
from langfence.retry import decide_next_step
from langfence.validation import (
    ValidationIssue,
    ValidationResult,
    validate_output,
    validate_provider_enforced_output,
)

ClientProfile: TypeAlias = Literal["vllm", "sglang", "openai", "litellm", "anthropic"]
ClientTransport: TypeAlias = Literal["openai", "anthropic"]

ANTHROPIC_VERSION = "2023-06-01"

_OPENAI_ENDPOINT = "/chat/completions"
_ANTHROPIC_ENDPOINT = "/messages"
_SYSTEM_ROLES = {"system", "developer"}
_ANTHROPIC_DEFAULT_MAX_TOKENS = 1024
_RETRYABLE_STATUS_CODES = frozenset({408, 429, 502, 503, 504})
_RETRY_BASE_DELAY = 0.5
_RETRY_MAX_DELAY = 8.0
_RETRY_AFTER_CAP = 30.0


@dataclass(frozen=True)
class ChatResult:
    text: str = field(repr=False)
    validation: ValidationResult = field(repr=False)
    attempts: int
    profile: ClientProfile
    transport: ClientTransport
    warnings: tuple[str, ...] = ()
    raw_response: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def parsed(self) -> Any | None:
        return self.validation.parsed

    @property
    def ok(self) -> bool:
        return self.validation.ok


class LangFenceClientError(Exception):
    """Base error for LangFence HTTP client failures."""


class LangFenceHTTPError(LangFenceClientError):
    def __init__(
        self,
        status_code: int,
        *,
        error_body: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(f"Provider returned HTTP status {status_code}.")
        self.status_code = status_code
        self.error_body = error_body
        self.retry_after = retry_after


class LangFenceResponseError(LangFenceClientError):
    """Raised when a provider response cannot be decoded or does not match its transport."""


@dataclass(frozen=True)
class _CompiledClientRequest:
    payload: dict[str, Any]
    warnings: tuple[str, ...] = ()


class LangFenceClient:
    """Synchronous chat client that compiles, sends, and validates output contracts.

    ``max_retries`` is a shared budget for validation-driven retries and
    retryable transport failures (HTTP 408/429/502/503/504 and network errors).
    ``max_tokens`` is only sent when set explicitly; the Anthropic transport,
    which requires the field, falls back to 1024. ``timeout`` is ignored when a
    pre-configured ``client`` is supplied — configure the timeout on that
    client instead.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        contract: OutputContract,
        provider: str | None = None,
        profile: str | None = None,
        transport: str | None = None,
        max_retries: int = 0,
        max_tokens: int | None = None,
        timeout: float | httpx.Timeout | None = 60.0,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        client: httpx.Client | None = None,
        include_error_body: bool = False,
        include_raw_response: bool = False,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to 0")
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")

        self.provider = provider
        self.profile = _normalize_profile(profile, provider)
        self.transport = _normalize_transport(transport, provider, self.profile)
        _validate_profile_transport(self.profile, self.transport)

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.contract = contract
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.headers = dict(headers or {})
        self.include_error_body = include_error_body
        self.include_raw_response = include_raw_response
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> LangFenceClient:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        **request_options: Any,
    ) -> ChatResult:
        if request_options.get("stream"):
            raise ValueError(
                "LangFenceClient does not support streaming responses (stream=True); "
                "validation needs the complete output."
            )

        warnings: list[str] = []
        repair_instructions: list[str] = []
        attempts_allowed = self.max_retries + 1
        last_result: ChatResult | None = None

        if self._provider_enforced_format():
            warnings.append(
                "Grammar/structural-tag constraints are enforced by the provider's constrained "
                "decoding and are not re-validated locally."
            )

        for attempt in range(1, attempts_allowed + 1):
            compiled = self._compile(messages, request_options, repair_instructions)
            warnings.extend(compiled.warnings)
            try:
                response_data = self._post(compiled.payload)
            except LangFenceHTTPError as exc:
                if attempt < attempts_allowed and exc.status_code in _RETRYABLE_STATUS_CODES:
                    _sleep_before_retry(attempt, exc.retry_after)
                    continue
                raise
            except LangFenceClientError as exc:
                if attempt < attempts_allowed and isinstance(
                    exc.__cause__, httpx.TransportError
                ):
                    _sleep_before_retry(attempt, None)
                    continue
                raise
            raw_text = self._extract_text(response_data)
            validation = self._validate(raw_text)
            last_result = ChatResult(
                text=validation.text,
                validation=validation,
                attempts=attempt,
                profile=self.profile,
                transport=self.transport,
                warnings=tuple(_unique(warnings)),
                raw_response=response_data if self.include_raw_response else None,
            )

            if validation.ok:
                return last_result

            # Re-prompting cannot fix issues the local validator is unable to
            # check at all; return immediately instead of burning retries.
            if _has_unfixable_errors(validation):
                return last_result

            decision = decide_next_step(validation, self.contract.language)
            if decision.action == "warn":
                return last_result

            if decision.action == "fail":
                if _has_only_language_errors(validation) or attempt >= attempts_allowed:
                    return last_result
                repair_instructions.append(_repair_instruction(validation.issues))
                continue

            if decision.action in {"retry", "repair"} and attempt < attempts_allowed:
                if decision.action == "repair":
                    repair_instructions.append(_repair_instruction(validation.issues))
                continue

            return last_result

        if last_result is None:
            raise LangFenceClientError("Unexpected retry loop termination.")
        return last_result

    def _compile(
        self,
        messages: Sequence[Mapping[str, Any]],
        request_options: Mapping[str, Any],
        repair_instructions: Sequence[str],
    ) -> _CompiledClientRequest:
        if self.transport == "openai":
            return self._compile_openai_transport(messages, request_options, repair_instructions)
        if self.transport == "anthropic":
            return self._compile_anthropic_transport(messages, request_options, repair_instructions)
        raise ValueError(f"Unsupported transport: {self.transport}")

    def _provider_enforced_format(self) -> bool:
        """True when the engine's constrained decoding enforces the format.

        Only vLLM/SGLang receive grammar/structural-tag request fields, and
        those constraints cannot be re-checked locally.
        """
        return self.profile in {"vllm", "sglang"} and isinstance(
            self.contract.format,
            GrammarConstraint | StructuralTagConstraint,
        )

    def _validate(self, text: str) -> ValidationResult:
        if self._provider_enforced_format():
            return validate_provider_enforced_output(text, self.contract)
        return validate_output(text, self.contract)

    def _compile_openai_transport(
        self,
        messages: Sequence[Mapping[str, Any]],
        request_options: Mapping[str, Any],
        repair_instructions: Sequence[str],
    ) -> _CompiledClientRequest:
        base_payload = dict(request_options)
        base_payload.setdefault("model", self.model)
        if self.max_tokens is not None:
            base_payload.setdefault("max_tokens", self.max_tokens)

        provider: str
        if self.profile in {"vllm", "sglang"}:
            provider = self.profile
        else:
            provider = "openai-compatible" if self.profile == "openai" else self.profile

        compiled = compile_request(
            provider,
            messages,
            self.contract,
            mode=RequestMode.OPENAI,
            base_payload=base_payload,
        )
        payload = compiled.payload
        warnings = list(compiled.warnings)

        for instruction in repair_instructions:
            _insert_openai_system_instruction(payload, instruction)

        return _CompiledClientRequest(payload=payload, warnings=tuple(warnings))

    def _compile_anthropic_transport(
        self,
        messages: Sequence[Mapping[str, Any]],
        request_options: Mapping[str, Any],
        repair_instructions: Sequence[str],
    ) -> _CompiledClientRequest:
        base_payload = dict(request_options)
        base_payload.setdefault("model", self.model)
        base_payload.setdefault(
            "max_tokens",
            self.max_tokens if self.max_tokens is not None else _ANTHROPIC_DEFAULT_MAX_TOKENS,
        )

        compiled = compile_request(
            "anthropic-compatible",
            messages,
            self.contract,
            mode=RequestMode.ANTHROPIC,
            base_payload=base_payload,
        )
        payload = compiled.payload
        for instruction in repair_instructions:
            existing = payload.get("system")
            payload["system"] = "\n\n".join(part for part in [existing, instruction] if part)
        return _CompiledClientRequest(payload=payload, warnings=compiled.warnings)

    def _post(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = dict(payload)
        # Adapters emit constraint fields under "extra_body" so compiled
        # payloads stay compatible with the OpenAI Python SDK, which merges
        # that key client-side. This client sends raw JSON, so merge it here —
        # otherwise engines silently ignore the unknown "extra_body" key and
        # the constraint never reaches constrained decoding.
        extra_body = body.pop("extra_body", None)
        if isinstance(extra_body, Mapping):
            for key, value in extra_body.items():
                body.setdefault(str(key), value)

        try:
            response = self._client.post(
                self.base_url + _endpoint_for_transport(self.transport),
                json=body,
                headers=self._request_headers(),
            )
        except httpx.HTTPError as exc:
            raise LangFenceClientError("Provider request failed.") from exc
        if response.status_code >= 400:
            raise LangFenceHTTPError(
                response.status_code,
                error_body=response.text if self.include_error_body else None,
                retry_after=_parse_retry_after(response.headers.get("retry-after")),
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise LangFenceResponseError("Provider returned a non-JSON response.") from exc
        if not isinstance(data, dict):
            raise LangFenceResponseError("Provider returned a non-object JSON response.")
        return data

    def _request_headers(self) -> dict[str, str]:
        headers = dict(self.headers)
        if self.api_key:
            if self.transport == "anthropic":
                headers.setdefault("x-api-key", self.api_key)
            else:
                headers.setdefault("Authorization", f"Bearer {self.api_key}")
        if self.transport == "anthropic":
            headers.setdefault("anthropic-version", ANTHROPIC_VERSION)
        return headers

    def _extract_text(self, response_data: Mapping[str, Any]) -> str:
        if self.transport == "openai":
            return _extract_openai_text(response_data)
        if self.transport == "anthropic":
            return _extract_anthropic_text(response_data)
        raise ValueError(f"Unsupported transport: {self.transport}")


def _normalize_profile(profile: str | None, provider: str | None) -> ClientProfile:
    value = _normalize_token(profile or provider or "openai")
    aliases: dict[str, ClientProfile] = {
        "vllm": "vllm",
        "sglang": "sglang",
        "openai": "openai",
        "openai-compatible": "openai",
        "compatible": "openai",
        "generic": "openai",
        "litellm": "litellm",
        "anthropic": "anthropic",
        "anthropic-compatible": "anthropic",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported client profile/provider: {profile or provider}") from exc


def _normalize_transport(
    transport: str | None,
    provider: str | None,
    profile: ClientProfile,
) -> ClientTransport:
    value = _normalize_token(transport) if transport is not None else ""
    if value:
        aliases: dict[str, ClientTransport] = {
            "openai": "openai",
            "openai-compatible": "openai",
            "chat-completions": "openai",
            "anthropic": "anthropic",
            "anthropic-compatible": "anthropic",
            "messages": "anthropic",
        }
        try:
            return aliases[value]
        except KeyError as exc:
            raise ValueError(f"Unsupported client transport: {transport}") from exc

    provider_value = _normalize_token(provider)
    if provider_value in {"anthropic", "anthropic-compatible"}:
        return "anthropic"
    if profile == "anthropic":
        return "anthropic"
    return "openai"


def _validate_profile_transport(profile: ClientProfile, transport: ClientTransport) -> None:
    if profile == "anthropic" and transport != "anthropic":
        raise ValueError("Anthropic profile requires transport='anthropic'")
    if transport == "anthropic" and profile != "anthropic":
        raise ValueError("Anthropic transport requires profile='anthropic'")


def _normalize_token(value: str | None) -> str:
    return (value or "").strip().lower().replace("_", "-")


def _endpoint_for_transport(transport: ClientTransport) -> str:
    if transport == "openai":
        return _OPENAI_ENDPOINT
    if transport == "anthropic":
        return _ANTHROPIC_ENDPOINT
    raise ValueError(f"Unsupported transport: {transport}")


def _insert_openai_system_instruction(payload: dict[str, Any], instruction: str) -> None:
    if not instruction:
        return
    messages = payload.setdefault("messages", [])
    if not isinstance(messages, list):
        raise TypeError("payload['messages'] must be a list")

    insert_at = 0
    while insert_at < len(messages):
        message = messages[insert_at]
        if not isinstance(message, Mapping) or message.get("role") not in _SYSTEM_ROLES:
            break
        insert_at += 1
    messages.insert(insert_at, {"role": "system", "content": instruction})


def _repair_instruction(issues: Sequence[ValidationIssue]) -> str:
    codes = ", ".join(_unique(issue.code for issue in issues if issue.severity == "error"))
    if not codes:
        codes = "validation.warning"
    return (
        "Previous response failed output contract validation. Retry with only a corrected "
        "final answer that satisfies the required format and language. Failed checks: "
        f"{codes}."
    )


def _has_only_language_errors(validation: ValidationResult) -> bool:
    errors = validation.errors
    return bool(errors) and all(issue.code.startswith("language.") for issue in errors)


def _has_unfixable_errors(validation: ValidationResult) -> bool:
    return any(issue.code.endswith(".validation_unavailable") for issue in validation.errors)


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    if seconds < 0:
        return None
    return seconds


def _sleep_before_retry(attempt: int, retry_after: float | None) -> None:
    if retry_after is not None:
        delay = min(retry_after, _RETRY_AFTER_CAP)
    else:
        delay = min(_RETRY_BASE_DELAY * (2 ** (attempt - 1)), _RETRY_MAX_DELAY)
    if delay > 0:
        time.sleep(delay)


def _extract_openai_text(response_data: Mapping[str, Any]) -> str:
    try:
        choice = response_data["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise LangFenceResponseError("OpenAI-compatible response is missing choices[0].") from exc

    if not isinstance(choice, Mapping):
        raise LangFenceResponseError("OpenAI-compatible response choice is not an object.")

    message = choice.get("message")
    if isinstance(message, Mapping) and "content" in message:
        return _content_to_text(message["content"])

    text = choice.get("text")
    if text is not None:
        return _content_to_text(text)

    raise LangFenceResponseError("OpenAI-compatible response is missing message content.")


def _extract_anthropic_text(response_data: Mapping[str, Any]) -> str:
    content = response_data.get("content")
    if content is None:
        raise LangFenceResponseError("Anthropic-compatible response is missing content.")
    return _content_to_text(content)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray, str)):
        parts: list[str] = []
        found_text = False
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    found_text = True
                    parts.append(text)
            elif isinstance(item, str):
                found_text = True
                parts.append(item)
        if found_text:
            return "".join(parts)
    raise LangFenceResponseError("Provider response did not contain text content.")


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values
