from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langfence.adapters.generic import (
    compile_anthropic_request,
    compile_openai_compatible_request,
)
from langfence.adapters.sglang import compile_sglang_request
from langfence.adapters.vllm import compile_vllm_request
from langfence.contracts import CompiledRequest, OutputContract, Provider, RequestMode


def compile_request(
    provider: str | Provider,
    messages: Sequence[Mapping[str, Any]] | None,
    contract: OutputContract,
    *,
    mode: str | RequestMode = RequestMode.OPENAI,
    base_payload: Mapping[str, Any] | None = None,
) -> CompiledRequest:
    normalized_provider = Provider(provider)
    normalized_mode = RequestMode(mode)
    payload: dict[str, Any] = dict(base_payload or {})
    if messages is not None:
        payload["messages"] = [dict(message) for message in messages]

    if normalized_provider is Provider.VLLM:
        return compile_vllm_request(payload, contract, mode=normalized_mode)
    if normalized_provider is Provider.SGLANG:
        return compile_sglang_request(payload, contract, mode=normalized_mode)
    if normalized_provider in {
        Provider.OPENAI,
        Provider.OPENAI_COMPATIBLE,
        Provider.LITELLM,
    }:
        return compile_openai_compatible_request(
            payload,
            contract,
            provider=normalized_provider,
            mode=normalized_mode,
        )
    if normalized_provider in {Provider.ANTHROPIC, Provider.ANTHROPIC_COMPATIBLE}:
        return compile_anthropic_request(
            payload,
            contract,
            provider=normalized_provider,
            mode=normalized_mode,
        )

    raise ValueError(f"Unsupported provider: {provider}")


__all__ = ["compile_request"]
