from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from langfence.adapters import compile_request
from langfence.constraints import OutputConstraint
from langfence.contracts import CompiledRequest, OutputContract, Provider, RequestMode
from langfence.language import LanguagePolicy
from langfence.validation import ValidationResult, validate_output

if TYPE_CHECKING:
    import httpx

    from langfence.clients import ChatResult, LangFenceClient


class LangFence:
    """Small facade for validating, compiling, and calling LLM output contracts."""

    def __init__(
        self,
        *,
        contract: OutputContract | None = None,
        format: OutputConstraint | None = None,
        language: LanguagePolicy | None = None,
        prompt_instruction: str | None = None,
    ) -> None:
        if contract is not None and (
            format is not None or language is not None or prompt_instruction is not None
        ):
            raise ValueError(
                "Pass either contract=... or individual format/language options, not both."
            )
        self.contract = contract or OutputContract(
            format=format,
            language=language,
            prompt_instruction=prompt_instruction,
        )

    def validate(self, text: str) -> ValidationResult:
        return validate_output(text, self.contract)

    def is_valid(self, text: str) -> bool:
        return self.validate(text).ok

    def compile(
        self,
        provider: str | Provider,
        messages: Sequence[Mapping[str, Any]] | None = None,
        *,
        mode: str | RequestMode | None = None,
        base_payload: Mapping[str, Any] | None = None,
    ) -> CompiledRequest:
        return compile_request(
            provider,
            messages,
            self.contract,
            mode=mode,
            base_payload=base_payload,
        )

    def client(
        self,
        *,
        base_url: str,
        model: str,
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
    ) -> LangFenceClient:
        from langfence.clients import LangFenceClient

        return LangFenceClient(
            provider=provider,
            profile=profile,
            transport=transport,
            base_url=base_url,
            model=model,
            contract=self.contract,
            max_retries=max_retries,
            max_tokens=max_tokens,
            timeout=timeout,
            api_key=api_key,
            headers=headers,
            client=client,
            include_error_body=include_error_body,
            include_raw_response=include_raw_response,
        )

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        base_url: str,
        model: str,
        provider: str | None = None,
        profile: str | None = None,
        transport: str | None = None,
        max_retries: int = 0,
        timeout: float | httpx.Timeout | None = 60.0,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        client: httpx.Client | None = None,
        include_error_body: bool = False,
        include_raw_response: bool = False,
        **request_options: Any,
    ) -> ChatResult:
        langfence_client = self.client(
            provider=provider,
            profile=profile,
            transport=transport,
            base_url=base_url,
            model=model,
            max_retries=max_retries,
            timeout=timeout,
            api_key=api_key,
            headers=headers,
            client=client,
            include_error_body=include_error_body,
            include_raw_response=include_raw_response,
        )
        try:
            return langfence_client.chat(messages, **request_options)
        finally:
            langfence_client.close()
