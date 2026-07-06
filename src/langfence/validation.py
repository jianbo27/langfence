from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from langfence.constraints import (
    ChoiceConstraint,
    GrammarConstraint,
    JsonSchemaConstraint,
    RegexConstraint,
    StructuralTagConstraint,
)
from langfence.contracts import OutputContract
from langfence.language import detect_language

IssueSeverity = Literal["warning", "error"]


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: IssueSeverity = "error"
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    issues: tuple[ValidationIssue, ...] = ()
    parsed: Any | None = None

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")


def validate_output(text: str, contract: OutputContract) -> ValidationResult:
    issues: list[ValidationIssue] = []
    parsed: Any | None = None

    if contract.format is not None:
        parsed = _validate_format(text, contract, issues)

    if contract.language is not None:
        _validate_language(text, parsed, contract, issues)

    return ValidationResult(
        ok=not any(issue.severity == "error" for issue in issues),
        issues=tuple(issues),
        parsed=parsed,
    )


def _validate_format(
    text: str,
    contract: OutputContract,
    issues: list[ValidationIssue],
) -> Any | None:
    constraint = contract.format
    if isinstance(constraint, JsonSchemaConstraint):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            issues.append(
                ValidationIssue(
                    code="json.invalid",
                    message=f"Output is not valid JSON: {exc.msg}",
                    path=str(exc.pos),
                )
            )
            return None

        validator = Draft202012Validator(constraint.schema)
        errors = sorted(validator.iter_errors(parsed), key=_jsonschema_error_key)
        for error in errors:
            issues.append(_jsonschema_issue(error))
        return parsed

    if isinstance(constraint, RegexConstraint):
        if re.fullmatch(constraint.pattern, text, flags=re.DOTALL) is None:
            issues.append(
                ValidationIssue(
                    code="regex.mismatch",
                    message="Output does not match the required regex.",
                    metadata={"pattern": constraint.pattern},
                )
            )
        return None

    if isinstance(constraint, ChoiceConstraint):
        if text not in constraint.choices:
            issues.append(
                ValidationIssue(
                    code="choice.invalid",
                    message="Output is not one of the allowed choices.",
                    metadata={"choices": list(constraint.choices)},
                )
            )
        return None

    if isinstance(constraint, GrammarConstraint):
        issues.append(
            ValidationIssue(
                code="grammar.validation_unavailable",
                message=(
                    "Grammar validation requires the provider backend; only request "
                    "compilation is checked."
                ),
                severity="warning",
            )
        )
        return None

    if isinstance(constraint, StructuralTagConstraint):
        issues.append(
            ValidationIssue(
                code="structural_tag.validation_unavailable",
                message=(
                    "Structural tag validation is provider-specific; only request "
                    "compilation is checked."
                ),
                severity="warning",
            )
        )
        return None

    raise TypeError(f"Unsupported constraint: {constraint!r}")


def _validate_language(
    text: str,
    parsed: Any | None,
    contract: OutputContract,
    issues: list[ValidationIssue],
) -> None:
    policy = contract.language
    if policy is None:
        return

    language_text = _language_detection_text(text, parsed)
    detection = detect_language(language_text, detector=policy.detector)
    severity: IssueSeverity = "warning" if policy.action == "warn" else "error"
    metadata = {
        "detected": detection.language,
        "confidence": detection.confidence,
        "detector": detection.detector,
        "action": policy.action,
    }

    if detection.confidence < policy.min_confidence:
        issues.append(
            ValidationIssue(
                code="language.low_confidence",
                message="Language detector confidence is below the policy threshold.",
                severity="warning",
                metadata=metadata,
            )
        )

    if policy.include and detection.language not in policy.include:
        issues.append(
            ValidationIssue(
                code="language.not_included",
                message="Detected language is not in the allowed language set.",
                severity=severity,
                metadata={**metadata, "include": list(policy.include)},
            )
        )

    excluded_language = _matched_excluded_language(detection, policy.exclude, policy.min_confidence)
    if excluded_language is not None:
        issues.append(
            ValidationIssue(
                code="language.excluded",
                message="Detected language is in the excluded language set.",
                severity=severity,
                metadata={
                    **metadata,
                    "exclude": list(policy.exclude),
                    "excluded": excluded_language,
                    "scores": detection.metadata,
                },
            )
        )


def _matched_excluded_language(
    detection: Any,
    excluded: tuple[str, ...],
    min_confidence: float,
) -> str | None:
    if not excluded:
        return None
    if detection.language in excluded:
        return str(detection.language)

    threshold = min(min_confidence, 0.20)
    for language in excluded:
        if detection.metadata.get(language, 0.0) >= threshold:
            return language
    return None


def _language_detection_text(text: str, parsed: Any | None) -> str:
    if parsed is None:
        return text

    values: list[str] = []
    _collect_language_values(parsed, path=(), values=values)
    return "\n".join(values) if values else text


def _collect_language_values(value: Any, *, path: tuple[str, ...], values: list[str]) -> None:
    if isinstance(value, str):
        if _is_language_metadata_value(value, path):
            return
        values.append(value)
        return

    if isinstance(value, dict):
        for key, item in value.items():
            _collect_language_values(item, path=(*path, str(key)), values=values)
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            _collect_language_values(item, path=(*path, str(index)), values=values)


def _is_language_metadata_value(value: str, path: tuple[str, ...]) -> bool:
    if not path:
        return False
    key = path[-1].lower()
    if key not in {"lang", "language", "locale", "language_code", "locale_code"}:
        return False
    return bool(re.fullmatch(r"[A-Za-z]{2,3}(?:[-_][A-Za-z]{2,4})?", value.strip()))


def _jsonschema_error_key(error: ValidationError) -> str:
    return ".".join(str(part) for part in error.absolute_path)


def _jsonschema_issue(error: ValidationError) -> ValidationIssue:
    path = ".".join(str(part) for part in error.absolute_path) or "$"
    return ValidationIssue(
        code="json_schema.invalid",
        message=error.message,
        path=path,
        metadata={"validator": error.validator},
    )
