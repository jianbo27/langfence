from langfence import (
    ChoiceConstraint,
    GrammarConstraint,
    JsonSchemaConstraint,
    LanguagePolicy,
    OutputContract,
    RegexConstraint,
    StructuralTagConstraint,
)
from langfence.validation import (
    _compiled_regex,
    _get_jsonschema_validator,
    extract_final_answer,
    validate_output,
)


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
    assert result.text == '{"answer": "ok"}'


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
    assert result.text == '{"answer": "ok"}'


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


def test_json_schema_issue_message_does_not_include_output_value() -> None:
    contract = OutputContract(
        format=JsonSchemaConstraint(
            schema={
                "type": "object",
                "properties": {"status": {"type": "string", "enum": ["approved"]}},
                "required": ["status"],
            }
        )
    )

    result = validate_output('{"status":"customer secret 123"}', contract)

    assert not result.ok
    assert "customer secret 123" not in result.issues[0].message
    assert result.issues[0].message == "Output failed JSON Schema validation at status."


def test_validate_choice_failure() -> None:
    result = validate_output(
        "maybe",
        OutputContract(format=ChoiceConstraint(["yes", "no"])),
    )

    assert not result.ok
    assert result.issues[0].code == "choice.invalid"


def test_grammar_validation_unavailable_is_error() -> None:
    result = validate_output(
        "anything",
        OutputContract(format=GrammarConstraint('root ::= "ok"')),
    )

    assert not result.ok
    assert result.issues[0].code == "grammar.validation_unavailable"


def test_structural_tag_validation_unavailable_is_error() -> None:
    result = validate_output(
        "<answer>ok</answer>",
        OutputContract(format=StructuralTagConstraint(spec={"begin": "<answer>"})),
    )

    assert not result.ok
    assert result.issues[0].code == "structural_tag.validation_unavailable"


def test_validate_regex_success_uses_cached_pattern() -> None:
    result = validate_output(
        "ok\nnext",
        OutputContract(format=RegexConstraint(r"ok\s+next")),
    )

    assert result.ok
    assert _compiled_regex(r"ok\s+next") is _compiled_regex(r"ok\s+next")


def test_json_schema_validator_cache_reuses_equivalent_schema() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    equivalent_schema = {
        "required": ["answer"],
        "properties": {"answer": {"type": "string"}},
        "type": "object",
    }

    assert _get_jsonschema_validator(schema) is _get_jsonschema_validator(equivalent_schema)


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
    assert extract_final_answer("```thinking\nprivate notes\n```\nFinal answer") == "Final answer"


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
