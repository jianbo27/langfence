from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from langfence.constraints import OutputConstraint
from langfence.language import LanguagePolicy


class Provider(str, Enum):
    VLLM = "vllm"
    SGLANG = "sglang"
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai-compatible"
    LITELLM = "litellm"
    ANTHROPIC = "anthropic"
    ANTHROPIC_COMPATIBLE = "anthropic-compatible"


class RequestMode(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    NATIVE = "native"


@dataclass(frozen=True)
class OutputContract:
    format: OutputConstraint | None = None
    language: LanguagePolicy | None = None
    prompt_instruction: str | None = None


@dataclass(frozen=True)
class CompiledRequest:
    provider: Provider
    mode: RequestMode
    payload: dict[str, Any]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    repair_prompt: str | None = None


@dataclass(frozen=True)
class ContractViolation:
    code: str
    message: str
    path: str | None = None
    severity: Literal["warning", "error"] = "error"
    metadata: dict[str, Any] = field(default_factory=dict)
