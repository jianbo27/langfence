import langfence
from langfence import ChoiceConstraint, LangFence, LanguagePolicy


def test_langfence_facade_validates_text() -> None:
    fence = LangFence(language=LanguagePolicy(include=["zh"], exclude=["en"], min_confidence=0.2))

    result = fence.validate("这是一个中文回答。")

    assert result.valid
    assert fence.is_valid("这是一个中文回答。")


def test_langfence_facade_compiles_provider_payload() -> None:
    fence = LangFence(format=ChoiceConstraint(["approved", "rejected"]))

    compiled = fence.compile(
        "vllm",
        [{"role": "user", "content": "Return approved or rejected."}],
        base_payload={"model": "local-model"},
    )

    assert compiled.payload["model"] == "local-model"
    assert compiled.payload["extra_body"]["structured_outputs"]["choice"] == [
        "approved",
        "rejected",
    ]


def test_langfence_facade_rejects_mixed_contract_options() -> None:
    import pytest

    with pytest.raises(ValueError, match="either contract"):
        LangFence(contract=langfence.OutputContract(), format=ChoiceConstraint(["ok"]))


def test_top_level_client_export_remains_available() -> None:
    assert langfence.LangFenceClient.__name__ == "LangFenceClient"
    assert langfence.__dict__["LangFenceClient"].__name__ == "LangFenceClient"


def test_top_level_compile_request_export_remains_available() -> None:
    assert langfence.compile_request.__name__ == "compile_request"
    assert langfence.__dict__["compile_request"].__name__ == "compile_request"


def test_top_level_validation_exports_remain_available() -> None:
    assert langfence.validate_output.__name__ == "validate_output"
    assert langfence.ValidationIssue.__name__ == "ValidationIssue"
    assert langfence.ValidationResult.__name__ == "ValidationResult"
    assert langfence.__dict__["validate_output"].__name__ == "validate_output"


def test_top_level_lazy_exports_are_discoverable() -> None:
    names = dir(langfence)

    assert "LangFence" in names
    assert "LangFenceClient" in names
    assert "compile_request" in names
    assert "validate_output" in names
