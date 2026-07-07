import pytest

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
    assert (
        result.issues[0].message
        == "Output failed JSON Schema validation at status: violates the 'enum' rule."
    )
    assert result.issues[0].metadata["validator_value"] == "['approved']"


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


def test_fenced_code_block_is_stripped_before_language_detection() -> None:
    # A Chinese answer that embeds an ASCII code block must not trip an
    # exclude=["en"] policy: the code fence is removed before detection.
    contract = OutputContract(
        language=LanguagePolicy(include=["zh"], exclude=["en"], min_confidence=0.2)
    )
    text = "这是一个中文回答，示例代码如下：\n```python\ndef add(a, b):\n    return a + b\n```"

    result = validate_output(text, contract)

    assert result.ok


def test_unterminated_fenced_code_block_is_stripped() -> None:
    contract = OutputContract(
        language=LanguagePolicy(include=["zh"], exclude=["en"], min_confidence=0.2)
    )
    text = "这是一个中文回答。\n```python\ndef add(a, b): return a + b"

    result = validate_output(text, contract)

    assert result.ok


def test_inline_code_is_stripped_before_language_detection() -> None:
    contract = OutputContract(
        language=LanguagePolicy(include=["zh"], exclude=["en"], min_confidence=0.2)
    )
    text = "这是一个中文回答，请调用 `calculate_total_amount` 函数。"

    result = validate_output(text, contract)

    assert result.ok


def test_urls_are_stripped_before_language_detection() -> None:
    contract = OutputContract(
        language=LanguagePolicy(include=["zh"], exclude=["en"], min_confidence=0.2)
    )
    text = "这是一个中文回答，详见 https://example.com/some/english/looking/path"

    result = validate_output(text, contract)

    assert result.ok


def test_include_low_signal_detection_downgrades_to_warning() -> None:
    # Numeric-only output carries no language signal. An include=["en"] policy
    # with action="fail" must not hard-fail on it: the violation is a warning
    # and the overall result stays ok.
    contract = OutputContract(
        language=LanguagePolicy(include=["en"], action="fail", min_confidence=0.75)
    )

    result = validate_output("42", contract)

    assert result.ok
    assert not result.errors
    codes = {issue.code for issue in result.warnings}
    assert "language.not_included" in codes


def test_include_high_signal_mismatch_is_error() -> None:
    # A confident Chinese answer against include=["en"] is a real violation.
    contract = OutputContract(
        language=LanguagePolicy(include=["en"], action="fail", min_confidence=0.2)
    )

    result = validate_output("这是一个中文回答。", contract)

    assert not result.ok
    assert any(issue.code == "language.not_included" for issue in result.errors)


def test_exclude_threshold_custom_value_is_honored() -> None:
    # A mostly-Chinese answer with a small English fragment. The English score
    # sits below the default 0.20 threshold but above a lowered 0.05 threshold,
    # so only the stricter policy reports an exclusion.
    text = "这是一个比较长的中文回答内容 ok"
    lenient = OutputContract(
        language=LanguagePolicy(exclude=["en"], min_confidence=0.1, exclude_threshold=0.20)
    )
    strict = OutputContract(
        language=LanguagePolicy(exclude=["en"], min_confidence=0.1, exclude_threshold=0.05)
    )

    assert validate_output(text, lenient).ok
    strict_result = validate_output(text, strict)
    assert not strict_result.ok
    assert any(issue.code == "language.excluded" for issue in strict_result.errors)


def test_regex_no_longer_matches_newline_without_dotall() -> None:
    result = validate_output("a\nb", OutputContract(format=RegexConstraint("a.b")))

    assert not result.ok
    assert result.issues[0].code == "regex.mismatch"


def test_regex_matches_newline_with_inline_dotall_flag() -> None:
    result = validate_output("a\nb", OutputContract(format=RegexConstraint("(?s)a.b")))

    assert result.ok


def test_regex_mismatch_metadata_carries_pattern_not_output() -> None:
    result = validate_output("nope", OutputContract(format=RegexConstraint(r"\d{3}")))

    assert not result.ok
    assert result.issues[0].metadata["pattern"] == r"\d{3}"
    assert "nope" not in result.issues[0].message


def test_json_schema_constraint_rejects_invalid_schema() -> None:
    with pytest.raises(ValueError, match="Invalid JSON Schema"):
        JsonSchemaConstraint(schema={"type": "objectt"})


def test_regex_constraint_rejects_uncompilable_pattern() -> None:
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        RegexConstraint(pattern="(")


def test_choice_constraint_rejects_empty_choices() -> None:
    with pytest.raises(ValueError, match="at least one choice"):
        ChoiceConstraint([])


def test_grammar_constraint_rejects_unknown_syntax() -> None:
    with pytest.raises(ValueError, match="Unsupported grammar syntax"):
        GrammarConstraint('root ::= "ok"', syntax="peg")  # type: ignore[arg-type]


def test_structural_tag_constraint_defaults_to_empty_spec() -> None:
    constraint = StructuralTagConstraint()

    assert constraint.spec == {}
    assert constraint.kind == "structural_tag"


def test_warn_action_downgrades_exclusion_error_to_warning() -> None:
    contract = OutputContract(
        language=LanguagePolicy(exclude=["en"], action="warn", min_confidence=0.2)
    )

    result = validate_output("This is a plainly English answer.", contract)

    assert result.ok
    assert any(issue.code == "language.excluded" for issue in result.warnings)
    assert not result.errors


def test_low_confidence_below_threshold_emits_warning() -> None:
    # A short mixed string detects with confidence under the 0.75 floor.
    contract = OutputContract(language=LanguagePolicy(include=["en"], min_confidence=0.75))

    result = validate_output("你好 hi", contract)

    assert any(issue.code == "language.low_confidence" for issue in result.warnings)
