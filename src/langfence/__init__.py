from typing import TYPE_CHECKING, Any

from langfence.constraints import (
    ChoiceConstraint,
    GrammarConstraint,
    JsonSchemaConstraint,
    RegexConstraint,
    StructuralTagConstraint,
)
from langfence.contracts import CompiledRequest, OutputContract, Provider, RequestMode
from langfence.language import LanguagePolicy

if TYPE_CHECKING:
    from langfence.adapters import compile_request
    from langfence.clients import (
        ChatResult,
        LangFenceClient,
        LangFenceClientError,
        LangFenceHTTPError,
        LangFenceResponseError,
    )
    from langfence.fence import LangFence
    from langfence.validation import ValidationIssue, ValidationResult, validate_output

_LAZY_EXPORTS = {
    "ChatResult",
    "LangFence",
    "LangFenceClient",
    "LangFenceClientError",
    "LangFenceHTTPError",
    "LangFenceResponseError",
    "ValidationIssue",
    "ValidationResult",
    "compile_request",
    "validate_output",
}

__all__ = [
    "ChatResult",
    "ChoiceConstraint",
    "CompiledRequest",
    "GrammarConstraint",
    "JsonSchemaConstraint",
    "LangFence",
    "LangFenceClient",
    "LangFenceClientError",
    "LangFenceHTTPError",
    "LangFenceResponseError",
    "LanguagePolicy",
    "OutputContract",
    "Provider",
    "RegexConstraint",
    "RequestMode",
    "StructuralTagConstraint",
    "ValidationIssue",
    "ValidationResult",
    "compile_request",
    "validate_output",
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name == "LangFence":
        from langfence.fence import LangFence

        globals()[name] = LangFence
        return LangFence

    if name == "compile_request":
        from langfence.adapters import compile_request

        globals()[name] = compile_request
        return compile_request

    if name in {"ValidationIssue", "ValidationResult", "validate_output"}:
        from langfence.validation import ValidationIssue, ValidationResult, validate_output

        validation_exports: dict[str, Any] = {
            "ValidationIssue": ValidationIssue,
            "ValidationResult": ValidationResult,
            "validate_output": validate_output,
        }
        value = validation_exports[name]
        globals()[name] = value
        return value

    from langfence.clients import (
        ChatResult,
        LangFenceClient,
        LangFenceClientError,
        LangFenceHTTPError,
        LangFenceResponseError,
    )

    client_exports: dict[str, Any] = {
        "ChatResult": ChatResult,
        "LangFenceClient": LangFenceClient,
        "LangFenceClientError": LangFenceClientError,
        "LangFenceHTTPError": LangFenceHTTPError,
        "LangFenceResponseError": LangFenceResponseError,
    }
    value = client_exports[name]
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
