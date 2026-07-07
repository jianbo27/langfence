from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

REDACTED = "[REDACTED]"

_SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "cookie",
    "password",
    "secret",
    "token",
)

_PROMPT_KEYS = {
    "content",
    "input",
    "inputs",
    "output",
    "prompt",
    "system",
    "text",
}


def redact_for_display(value: Any) -> Any:
    return _redact(value, path=())


def _redact(value: Any, *, path: tuple[str, ...]) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = key.lower()
            next_path = (*path, normalized)
            if _is_secret_key(normalized) or _is_prompt_value_key(normalized, path):
                redacted[key] = REDACTED
            else:
                redacted[key] = _redact(item, path=next_path)
        return redacted

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_redact(item, path=path) for item in value]

    return value


def _is_secret_key(key: str) -> bool:
    return any(part in key for part in _SECRET_KEY_PARTS)


def _is_prompt_value_key(key: str, path: tuple[str, ...]) -> bool:
    if key not in _PROMPT_KEYS:
        return False

    if "messages" in path:
        return True

    if "message" in path or "choices" in path:
        return True

    if not path:
        return True

    parent = path[-1]
    return parent in {"request", "payload", "body", "response"}
