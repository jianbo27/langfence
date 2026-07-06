from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from langfence.constraints import (
    ChoiceConstraint,
    GrammarConstraint,
    JsonSchemaConstraint,
    OutputConstraint,
    RegexConstraint,
    StructuralTagConstraint,
)
from langfence.contracts import OutputContract
from langfence.language import LanguagePolicy


def load_contract(path: str | Path) -> OutputContract:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("Contract file must contain a YAML mapping")
    return contract_from_dict(data)


def contract_from_dict(data: dict[str, Any]) -> OutputContract:
    format_data = data.get("format")
    language_data = data.get("language")

    return OutputContract(
        format=_constraint_from_dict(format_data) if format_data else None,
        language=_language_from_dict(language_data) if language_data else None,
        prompt_instruction=data.get("prompt_instruction"),
    )


def _constraint_from_dict(data: dict[str, Any]) -> OutputConstraint:
    if not isinstance(data, dict):
        raise ValueError("format must be a mapping")

    kind = data.get("type") or data.get("kind")
    if kind == "json_schema":
        schema = data.get("schema")
        if not isinstance(schema, dict):
            raise ValueError("json_schema format requires a schema mapping")
        return JsonSchemaConstraint(
            schema=schema,
            name=str(data.get("name", "output")),
            description=data.get("description"),
            strict=bool(data.get("strict", True)),
        )
    if kind == "regex":
        return RegexConstraint(pattern=str(data["pattern"]))
    if kind == "choice":
        choices = data.get("choices")
        if not isinstance(choices, list):
            raise ValueError("choice format requires a choices list")
        return ChoiceConstraint([str(choice) for choice in choices])
    if kind in {"grammar", "ebnf"}:
        return GrammarConstraint(
            grammar=str(data["grammar"]),
            syntax=str(data.get("syntax", "ebnf")),  # type: ignore[arg-type]
        )
    if kind == "structural_tag":
        spec = data.get("spec", {})
        if not isinstance(spec, dict):
            raise ValueError("structural_tag format requires a spec mapping")
        return StructuralTagConstraint(spec=spec)

    raise ValueError(f"Unsupported format type: {kind}")


def _language_from_dict(data: dict[str, Any]) -> LanguagePolicy:
    if not isinstance(data, dict):
        raise ValueError("language must be a mapping")

    return LanguagePolicy(
        include=tuple(str(item) for item in data.get("include", ())),
        exclude=tuple(str(item) for item in data.get("exclude", ())),
        action=data.get("action", "fail"),
        min_confidence=float(data.get("min_confidence", 0.75)),
        detector=str(data.get("detector", "heuristic")),
    )


def contract_to_dict(contract: OutputContract) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if contract.prompt_instruction:
        data["prompt_instruction"] = contract.prompt_instruction
    if contract.format:
        data["format"] = _constraint_to_dict(contract.format)
    if contract.language:
        data["language"] = {
            "include": list(contract.language.include),
            "exclude": list(contract.language.exclude),
            "action": contract.language.action,
            "min_confidence": contract.language.min_confidence,
            "detector": contract.language.detector,
        }
    return data


def _constraint_to_dict(constraint: object) -> dict[str, Any]:
    if isinstance(constraint, JsonSchemaConstraint):
        return {
            "type": "json_schema",
            "name": constraint.name,
            "description": constraint.description,
            "strict": constraint.strict,
            "schema": constraint.schema,
        }
    if isinstance(constraint, RegexConstraint):
        return {"type": "regex", "pattern": constraint.pattern}
    if isinstance(constraint, ChoiceConstraint):
        return {"type": "choice", "choices": list(constraint.choices)}
    if isinstance(constraint, GrammarConstraint):
        return {"type": "grammar", "grammar": constraint.grammar, "syntax": constraint.syntax}
    if isinstance(constraint, StructuralTagConstraint):
        return {"type": "structural_tag", "spec": constraint.spec}
    raise TypeError(f"Unsupported constraint: {constraint!r}")
