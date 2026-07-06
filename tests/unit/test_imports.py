import langfence


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

    assert "LangFenceClient" in names
    assert "compile_request" in names
    assert "validate_output" in names
