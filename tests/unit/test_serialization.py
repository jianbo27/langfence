from __future__ import annotations

from pathlib import Path

import pytest

from langfence.constraints import (
    ChoiceConstraint,
    GrammarConstraint,
    JsonSchemaConstraint,
    RegexConstraint,
    StructuralTagConstraint,
)
from langfence.contracts import OutputContract
from langfence.language import LanguagePolicy
from langfence.serialization import (
    contract_from_dict,
    contract_to_dict,
    load_contract,
)


def test_load_contract_yaml() -> None:
    contract = load_contract("examples/contract.zh.yaml")

    assert isinstance(contract.format, JsonSchemaConstraint)
    assert contract.format.name == "localized_answer"
    assert contract.language is not None
    assert contract.language.include == ("zh",)


def test_scalar_include_is_a_single_code_not_characters() -> None:
    contract = contract_from_dict({"language": {"include": "zh"}})

    assert contract.language is not None
    assert contract.language.include == ("zh",)


def test_scalar_exclude_is_a_single_code_not_characters() -> None:
    contract = contract_from_dict({"language": {"exclude": "en"}})

    assert contract.language is not None
    assert contract.language.exclude == ("en",)


def test_language_include_rejects_non_string_entries() -> None:
    with pytest.raises(ValueError, match="language.include entries must be strings"):
        contract_from_dict({"language": {"include": ["zh", 42]}})


def test_language_exclude_rejects_non_string_entries() -> None:
    with pytest.raises(ValueError, match="language.exclude entries must be strings"):
        contract_from_dict({"language": {"exclude": [None]}})


def test_language_include_rejects_non_list_non_string() -> None:
    with pytest.raises(ValueError, match="must be a string or a list of strings"):
        contract_from_dict({"language": {"include": {"zh": True}}})


def test_regex_format_requires_pattern_string() -> None:
    with pytest.raises(ValueError, match="regex format requires a pattern string"):
        contract_from_dict({"format": {"type": "regex"}})


def test_grammar_format_requires_grammar_string() -> None:
    with pytest.raises(ValueError, match="grammar format requires a grammar string"):
        contract_from_dict({"format": {"type": "grammar"}})


def test_json_schema_format_requires_schema_mapping() -> None:
    with pytest.raises(ValueError, match="json_schema format requires a schema mapping"):
        contract_from_dict({"format": {"type": "json_schema"}})


def test_choice_format_requires_non_empty_choices() -> None:
    with pytest.raises(ValueError, match="non-empty choices list"):
        contract_from_dict({"format": {"type": "choice", "choices": []}})


def test_unsupported_format_type_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported format type"):
        contract_from_dict({"format": {"type": "mystery"}})


def test_exclude_threshold_round_trips_through_dict() -> None:
    contract = contract_from_dict(
        {"language": {"exclude": ["en"], "exclude_threshold": 0.42}}
    )

    assert contract.language is not None
    assert contract.language.exclude_threshold == 0.42
    assert contract_to_dict(contract)["language"]["exclude_threshold"] == 0.42


def test_utf8_prompt_instruction_round_trips_through_file(tmp_path: Path) -> None:
    instruction = "请用中文回答，并保持礼貌。"
    contract_file = tmp_path / "contract.yaml"
    contract_file.write_text(
        f'prompt_instruction: "{instruction}"\n',
        encoding="utf-8",
    )

    contract = load_contract(contract_file)

    assert contract.prompt_instruction == instruction


def test_load_contract_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    contract_file = tmp_path / "contract.yaml"
    contract_file.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_contract(contract_file)


def test_load_contract_empty_file_yields_empty_contract(tmp_path: Path) -> None:
    contract_file = tmp_path / "contract.yaml"
    contract_file.write_text("", encoding="utf-8")

    contract = load_contract(contract_file)

    assert contract == OutputContract()


@pytest.mark.parametrize(
    "constraint",
    [
        JsonSchemaConstraint(
            schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
            name="answer",
            description="an answer",
            strict=True,
        ),
        RegexConstraint(r"\d{3}-\d{4}"),
        ChoiceConstraint(["approved", "rejected"]),
        GrammarConstraint('root ::= "ok"', syntax="ebnf"),
        StructuralTagConstraint(spec={"begin": "<a>", "end": "</a>"}),
    ],
    ids=["json_schema", "regex", "choice", "grammar", "structural_tag"],
)
def test_constraint_round_trip_preserves_equality(constraint: object) -> None:
    contract = OutputContract(format=constraint)  # type: ignore[arg-type]

    restored = contract_from_dict(contract_to_dict(contract))

    assert restored.format == constraint


def test_full_contract_round_trip_equality() -> None:
    contract = OutputContract(
        format=JsonSchemaConstraint(
            schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
            name="localized_answer",
        ),
        language=LanguagePolicy(
            include=["zh"],
            exclude=["en", "ja"],
            action="repair",
            min_confidence=0.6,
            detector="heuristic",
            exclude_threshold=0.15,
        ),
        prompt_instruction="请用中文回答。",
    )

    restored = contract_from_dict(contract_to_dict(contract))

    assert restored == contract


def test_language_policy_round_trip_equality() -> None:
    contract = OutputContract(
        language=LanguagePolicy(
            include=["en"],
            exclude=["zh"],
            action="warn",
            min_confidence=0.9,
            exclude_threshold=0.33,
        )
    )

    restored = contract_from_dict(contract_to_dict(contract))

    assert restored.language == contract.language
