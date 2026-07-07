from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
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

_VISIBLE_REASONING_BLOCK_RE = re.compile(
    r"\A\s*(?:"
    r"<(?P<tag>think|thinking|reasoning)>\s*.*?\s*</(?P=tag)>"
    r"|"
    r"```(?:think|thinking|reasoning)\s*.*?\s*```"
    r")\s*",
    flags=re.DOTALL | re.IGNORECASE,
)
_VISIBLE_REASONING_PREFIXES = ("<think", "<reasoning", "```think", "```reasoning")


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
    parsed: Any | None = field(default=None, repr=False)
    text: str = field(default="", repr=False)

    @property
    def valid(self) -> bool:
        return self.ok

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")


def validate_output(text: str, contract: OutputContract) -> ValidationResult:
    return _validate_output(text, contract, skip_format_validation=False)


def validate_provider_enforced_output(text: str, contract: OutputContract) -> ValidationResult:
    return _validate_output(text, contract, skip_format_validation=True)


def _validate_output(
    text: str,
    contract: OutputContract,
    *,
    skip_format_validation: bool,
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    parsed: Any | None = None
    validation_text = extract_final_answer(text)

    if contract.format is not None and not skip_format_validation:
        parsed = _validate_format(validation_text, contract, issues)

    if contract.language is not None:
        _validate_language(validation_text, parsed, contract, issues)

    return ValidationResult(
        ok=not any(issue.severity == "error" for issue in issues),
        issues=tuple(issues),
        parsed=parsed,
        text=validation_text,
    )


def extract_final_answer(text: str) -> str:
    """Remove leading visible reasoning blocks without changing provider requests."""
    previous = text
    while _has_visible_reasoning_prefix(previous):
        stripped = _VISIBLE_REASONING_BLOCK_RE.sub("", previous, count=1)
        if stripped == previous:
            break
        previous = stripped
    return previous.strip()


def _has_visible_reasoning_prefix(text: str) -> bool:
    index = 0
    while index < len(text) and text[index].isspace():
        index += 1
    return text[index : index + 16].lower().startswith(_VISIBLE_REASONING_PREFIXES)


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

        validator = _get_jsonschema_validator(constraint.schema)
        if validator.is_valid(parsed):
            return parsed

        errors = list(validator.iter_errors(parsed))
        if len(errors) > 1:
            errors.sort(key=_jsonschema_error_key)
        for error in errors:
            issues.append(_jsonschema_issue(error))
        return parsed

    if isinstance(constraint, RegexConstraint):
        if _compiled_regex(constraint.pattern).fullmatch(text) is None:
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
                    "Grammar validation requires provider-side enforcement and is unavailable "
                    "for local post-validation."
                ),
                severity="error",
            )
        )
        return None

    if isinstance(constraint, StructuralTagConstraint):
        issues.append(
            ValidationIssue(
                code="structural_tag.validation_unavailable",
                message=(
                    "Structural tag validation requires provider-side enforcement and is "
                    "unavailable for local post-validation."
                ),
                severity="error",
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


def _get_jsonschema_validator(schema: dict[str, Any]) -> Draft202012Validator:
    try:
        schema_key = _jsonschema_cache_key(schema)
    except (TypeError, ValueError):
        return Draft202012Validator(schema)
    return _cached_jsonschema_validator(schema_key)


def _jsonschema_cache_key(schema: dict[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


@lru_cache(maxsize=256)
def _cached_jsonschema_validator(schema_key: str) -> Draft202012Validator:
    schema = json.loads(schema_key)
    if not isinstance(schema, dict):
        raise TypeError("JSON Schema cache key did not decode to an object.")
    return Draft202012Validator(schema)


@lru_cache(maxsize=512)
def _compiled_regex(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, flags=re.DOTALL)


def _jsonschema_error_key(error: ValidationError) -> str:
    return ".".join(str(part) for part in error.absolute_path)


def _jsonschema_issue(error: ValidationError) -> ValidationIssue:
    path = ".".join(str(part) for part in error.absolute_path) or "$"
    return ValidationIssue(
        code="json_schema.invalid",
        message=f"Output failed JSON Schema validation at {path}.",
        path=path,
        metadata={"validator": error.validator},
    )
