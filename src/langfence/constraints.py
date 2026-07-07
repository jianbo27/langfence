from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


@dataclass(frozen=True)
class JsonSchemaConstraint:
    schema: dict[str, Any]
    name: str = "output"
    description: str | None = None
    strict: bool = True
    kind: Literal["json_schema"] = "json_schema"

    def __post_init__(self) -> None:
        try:
            Draft202012Validator.check_schema(self.schema)
        except SchemaError as exc:
            raise ValueError(f"Invalid JSON Schema: {exc.message}") from exc


@dataclass(frozen=True)
class RegexConstraint:
    pattern: str
    kind: Literal["regex"] = "regex"

    def __post_init__(self) -> None:
        try:
            re.compile(self.pattern)
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern: {exc}") from exc


@dataclass(frozen=True)
class ChoiceConstraint:
    choices: tuple[str, ...]
    kind: Literal["choice"] = "choice"

    def __init__(self, choices: list[str] | tuple[str, ...]) -> None:
        if not choices:
            raise ValueError("ChoiceConstraint requires at least one choice")
        object.__setattr__(self, "choices", tuple(choices))
        object.__setattr__(self, "kind", "choice")


@dataclass(frozen=True)
class GrammarConstraint:
    grammar: str
    syntax: Literal["ebnf", "lark"] = "ebnf"
    kind: Literal["grammar"] = "grammar"

    def __post_init__(self) -> None:
        if self.syntax not in ("ebnf", "lark"):
            raise ValueError(
                f"Unsupported grammar syntax: {self.syntax!r} (expected 'ebnf' or 'lark')"
            )


@dataclass(frozen=True)
class StructuralTagConstraint:
    spec: dict[str, Any] = field(default_factory=dict)
    kind: Literal["structural_tag"] = "structural_tag"


OutputConstraint: TypeAlias = (
    JsonSchemaConstraint
    | RegexConstraint
    | ChoiceConstraint
    | GrammarConstraint
    | StructuralTagConstraint
)
