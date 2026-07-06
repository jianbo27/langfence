from langfence import (
    ChoiceConstraint,
    JsonSchemaConstraint,
    LanguagePolicy,
    OutputContract,
)
from langfence.validation import extract_final_answer, validate_output


def test_validate_json_schema_success() -> None:
    contract = OutputContract(
        format=JsonSchemaConstraint(
            schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            }
        )
    )

    result = validate_output('{"answer": "ok"}', contract)

    assert result.ok
    assert result.parsed == {"answer": "ok"}


def test_validate_json_schema_ignores_leading_visible_reasoning() -> None:
    contract = OutputContract(
        format=JsonSchemaConstraint(
            schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            }
        )
    )

    result = validate_output(
        '<think>use any internal language here</think>\n{"answer": "ok"}',
        contract,
    )

    assert result.ok
    assert result.parsed == {"answer": "ok"}


def test_validate_json_schema_failure() -> None:
    contract = OutputContract(
        format=JsonSchemaConstraint(
            schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            }
        )
    )

    result = validate_output("{}", contract)

    assert not result.ok
    assert result.issues[0].code == "json_schema.invalid"


def test_validate_choice_failure() -> None:
    result = validate_output(
        "maybe",
        OutputContract(format=ChoiceConstraint(["yes", "no"])),
    )

    assert not result.ok
    assert result.issues[0].code == "choice.invalid"


def test_validate_language_exclusion_failure() -> None:
    contract = OutputContract(language=LanguagePolicy(exclude=["en"], min_confidence=0.2))

    result = validate_output("This is an English answer.", contract)

    assert not result.ok
    assert result.issues[0].code == "language.excluded"


def test_validate_language_exclusion_catches_mixed_language_leak() -> None:
    contract = OutputContract(
        language=LanguagePolicy(include=["zh"], exclude=["en"], min_confidence=0.1)
    )

    result = validate_output("这是中文回答 with an English leak", contract)

    assert not result.ok
    assert any(issue.code == "language.excluded" for issue in result.issues)


def test_validate_language_include_success_for_chinese_text() -> None:
    contract = OutputContract(language=LanguagePolicy(include=["zh"], min_confidence=0.2))

    result = validate_output("这是一个中文回答。", contract)

    assert result.ok


def test_validate_language_ignores_leading_visible_reasoning_block() -> None:
    contract = OutputContract(
        language=LanguagePolicy(include=["zh"], exclude=["en"], min_confidence=0.2)
    )

    result = validate_output(
        "<think>I may reason internally in English.</think>\n这是一个中文回答。",
        contract,
    )

    assert result.ok


def test_extract_final_answer_removes_common_reasoning_fences() -> None:
    assert (
        extract_final_answer("```thinking\nprivate notes\n```\nFinal answer")
        == "Final answer"
    )


def test_validate_language_uses_json_string_values_not_keys() -> None:
    contract = OutputContract(
        format=JsonSchemaConstraint(
            schema={
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "language": {"type": "string", "enum": ["zh"]},
                },
                "required": ["answer", "language"],
            }
        ),
        language=LanguagePolicy(include=["zh"], exclude=["en"], min_confidence=0.2),
    )

    result = validate_output('{"answer":"这是中文回答。","language":"zh"}', contract)

    assert result.ok
