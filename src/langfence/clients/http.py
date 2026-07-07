from __future__ import annotations

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

_OPENAI_ENDPOINT = "/chat/completions"
_ANTHROPIC_ENDPOINT = "/messages"
_SYSTEM_ROLES = {"system", "developer"}


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
    ) -> None:
        super().__init__(f"Provider returned HTTP status {status_code}.")
        self.status_code = status_code
        self.error_body = error_body


class LangFenceResponseError(LangFenceClientError):
    """Raised when a provider response cannot be decoded or does not match its transport."""


@dataclass(frozen=True)
class _CompiledClientRequest:
    payload: dict[str, Any]
    warnings: tuple[str, ...] = ()


class LangFenceClient:
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
        max_tokens: int = 1024,
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
        warnings: list[str] = []
        repair_instructions: list[str] = []
        attempts_allowed = self.max_retries + 1
        last_result: ChatResult | None = None

        for attempt in range(1, attempts_allowed + 1):
            compiled = self._compile(messages, request_options, repair_instructions)
            warnings.extend(compiled.warnings)
            response_data = self._post(compiled.payload)
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

    def _validate(self, text: str) -> ValidationResult:
        if self.profile in {"vllm", "sglang"} and isinstance(
            self.contract.format,
            GrammarConstraint | StructuralTagConstraint,
        ):
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
        base_payload.setdefault("max_tokens", self.max_tokens)

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
        try:
            response = self._client.post(
                self.base_url + _endpoint_for_transport(self.transport),
                json=payload,
                headers=self._request_headers(),
            )
        except httpx.HTTPError as exc:
            raise LangFenceClientError("Provider request failed.") from exc
        if response.status_code >= 400:
            raise LangFenceHTTPError(
                response.status_code,
                error_body=response.text if self.include_error_body else None,
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
            headers.setdefault("anthropic-version", "2023-06-01")
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
