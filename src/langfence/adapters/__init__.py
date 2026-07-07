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
    mode: str | RequestMode | None = None,
    base_payload: Mapping[str, Any] | None = None,
) -> CompiledRequest:
    normalized_provider = _normalize_provider(provider)
    normalized_mode = _normalize_mode(mode, normalized_provider)
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


def _normalize_provider(provider: str | Provider) -> Provider:
    if isinstance(provider, Provider):
        return provider

    value = provider.strip().lower().replace("_", "-")
    aliases = {
        "vllm": Provider.VLLM,
        "sglang": Provider.SGLANG,
        "openai": Provider.OPENAI,
        "openai-compatible": Provider.OPENAI_COMPATIBLE,
        "compatible": Provider.OPENAI_COMPATIBLE,
        "generic": Provider.OPENAI_COMPATIBLE,
        "litellm": Provider.LITELLM,
        "anthropic": Provider.ANTHROPIC,
        "anthropic-compatible": Provider.ANTHROPIC_COMPATIBLE,
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider: {provider}") from exc


def _normalize_mode(mode: str | RequestMode | None, provider: Provider) -> RequestMode:
    if isinstance(mode, RequestMode):
        return mode
    if mode is None:
        if provider in {Provider.ANTHROPIC, Provider.ANTHROPIC_COMPATIBLE}:
            return RequestMode.ANTHROPIC
        return RequestMode.OPENAI

    value = mode.strip().lower().replace("_", "-")
    try:
        return RequestMode(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported request mode: {mode}") from exc


__all__ = ["compile_request"]
