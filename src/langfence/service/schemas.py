from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CompileRequestBody(BaseModel):
    provider: str
    messages: list[dict[str, Any]] | None = None
    contract: dict[str, Any]
    mode: str | None = None
    base_payload: dict[str, Any] = Field(default_factory=dict)
    redact: bool = True


class ValidateRequestBody(BaseModel):
    contract: dict[str, Any]
    output: str
    redact: bool = True
