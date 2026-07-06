from langfence import (
    ChoiceConstraint,
    GrammarConstraint,
    JsonSchemaConstraint,
    LanguagePolicy,
    OutputContract,
    RegexConstraint,
    compile_request,
)


def test_vllm_json_schema_openai_response_format() -> None:
    contract = OutputContract(
        format=JsonSchemaConstraint(name="answer", schema={"type": "object"})
    )

    compiled = compile_request(
        "vllm",
        [{"role": "user", "content": "hi"}],
        contract,
        base_payload={"model": "m"},
    )

    assert compiled.payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {"type": "object"},
            "strict": True,
        },
    }


def test_vllm_regex_openai_uses_structured_outputs() -> None:
    compiled = compile_request(
        "vllm",
        None,
        OutputContract(format=RegexConstraint(r"[0-9]+")),
    )

    assert compiled.payload["extra_body"]["structured_outputs"]["regex"] == r"[0-9]+"


def test_vllm_choice_openai_is_native_choice() -> None:
    compiled = compile_request(
        "vllm",
        None,
        OutputContract(format=ChoiceConstraint(["yes", "no"])),
    )

    assert compiled.payload["extra_body"]["structured_outputs"]["choice"] == ["yes", "no"]


def test_sglang_json_schema_openai_response_format() -> None:
    compiled = compile_request(
        "sglang",
        [{"role": "user", "content": "hi"}],
        OutputContract(format=JsonSchemaConstraint(name="answer", schema={"type": "object"})),
    )

    assert compiled.payload["response_format"]["type"] == "json_schema"
    assert compiled.payload["response_format"]["json_schema"]["name"] == "answer"


def test_sglang_regex_openai_uses_extra_body_regex() -> None:
    compiled = compile_request(
        "sglang",
        None,
        OutputContract(format=RegexConstraint(r"[0-9]+")),
    )

    assert compiled.payload["extra_body"]["regex"] == r"[0-9]+"


def test_sglang_choice_falls_back_to_regex() -> None:
    compiled = compile_request(
        "sglang",
        None,
        OutputContract(format=ChoiceConstraint(["yes", "no"])),
    )

    assert compiled.payload["extra_body"]["regex"] == "^(?:yes|no)$"
    assert "compiled choices to regex" in compiled.warnings[0]


def test_sglang_native_grammar_uses_ebnf_sampling_param() -> None:
    compiled = compile_request(
        "sglang",
        None,
        OutputContract(format=GrammarConstraint('root ::= "ok"')),
        mode="native",
    )

    assert compiled.payload["sampling_params"]["ebnf"] == 'root ::= "ok"'


def test_openai_compatible_regex_uses_prompt_guidance_not_private_fields() -> None:
    compiled = compile_request(
        "openai-compatible",
        [{"role": "user", "content": "hi"}],
        OutputContract(format=RegexConstraint(r"ok|no")),
        base_payload={"model": "m"},
    )

    assert compiled.payload["model"] == "m"
    assert "response_format" not in compiled.payload
    assert "extra_body" not in compiled.payload
    assert compiled.payload["messages"][0]["role"] == "system"
    assert "regex" in compiled.payload["messages"][0]["content"]
    assert any("post-validation only" in warning for warning in compiled.warnings)


def test_litellm_json_schema_uses_standard_response_format() -> None:
    compiled = compile_request(
        "litellm",
        [{"role": "user", "content": "hi"}],
        OutputContract(format=JsonSchemaConstraint(name="answer", schema={"type": "object"})),
        base_payload={"model": "m"},
    )

    assert compiled.payload["response_format"]["type"] == "json_schema"
    assert "extra_body" not in compiled.payload
    assert "sampling_params" not in compiled.payload


def test_anthropic_compatible_uses_messages_payload_and_system_guidance() -> None:
    compiled = compile_request(
        "anthropic-compatible",
        [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hi"},
        ],
        OutputContract(
            format=RegexConstraint(r"ok|no"),
            language=LanguagePolicy(include=["en"], exclude=["zh"], min_confidence=0.2),
        ),
        mode="anthropic",
        base_payload={"model": "claude-compatible"},
    )

    assert compiled.payload["model"] == "claude-compatible"
    assert compiled.payload["max_tokens"] == 1024
    assert compiled.payload["messages"] == [{"role": "user", "content": "hi"}]
    assert "Be terse." in compiled.payload["system"]
    assert "regex" in compiled.payload["system"]
    assert "Use only these natural languages: en." in compiled.payload["system"]
    assert "response_format" not in compiled.payload
    assert "extra_body" not in compiled.payload
