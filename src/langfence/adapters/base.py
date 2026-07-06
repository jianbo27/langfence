from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from langfence.contracts import OutputContract
from langfence.language import language_instruction


def clone_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(payload))


def add_language_instruction(payload: dict[str, Any], contract: OutputContract) -> None:
    instructions: list[str] = []
    if contract.prompt_instruction:
        instructions.append(contract.prompt_instruction)
    if contract.language:
        instruction = language_instruction(contract.language)
        if instruction:
            instructions.append(instruction)
    if not instructions:
        return

    message = {"role": "system", "content": "\n".join(instructions)}
    messages = payload.setdefault("messages", [])
    if not isinstance(messages, list):
        raise TypeError(
            "payload['messages'] must be a list when language instructions are injected"
        )
    messages.insert(0, message)
