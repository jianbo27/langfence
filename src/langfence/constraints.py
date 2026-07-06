from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias


@dataclass(frozen=True)
class JsonSchemaConstraint:
    schema: dict[str, Any]
    name: str = "output"
    description: str | None = None
    strict: bool = True
    kind: Literal["json_schema"] = "json_schema"


@dataclass(frozen=True)
class RegexConstraint:
    pattern: str
    kind: Literal["regex"] = "regex"


@dataclass(frozen=True)
class ChoiceConstraint:
    choices: tuple[str, ...]
    kind: Literal["choice"] = "choice"

    def __init__(self, choices: list[str] | tuple[str, ...]) -> None:
        object.__setattr__(self, "choices", tuple(choices))
        object.__setattr__(self, "kind", "choice")


@dataclass(frozen=True)
class GrammarConstraint:
    grammar: str
    syntax: Literal["ebnf", "lark"] = "ebnf"
    kind: Literal["grammar"] = "grammar"


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
