from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from langfence.language import LanguagePolicy
from langfence.validation import ValidationResult

RetryDecision = Literal["accept", "warn", "retry", "repair", "fail"]


@dataclass(frozen=True)
class PolicyDecision:
    action: RetryDecision
    reason: str | None = None


def decide_next_step(
    result: ValidationResult,
    language_policy: LanguagePolicy | None,
) -> PolicyDecision:
    if result.ok:
        return PolicyDecision("accept")

    if language_policy is None:
        return PolicyDecision("fail", "Output violates the format contract.")

    errors = result.errors
    language_issues = [issue for issue in errors if issue.code.startswith("language.")]
    if not language_issues or len(language_issues) != len(errors):
        return PolicyDecision("fail", "Output violates the format contract.")

    if language_policy.action == "warn":
        return PolicyDecision("warn", "Output violates language policy.")
    if language_policy.action == "retry":
        return PolicyDecision("retry", "Output violates language policy.")
    if language_policy.action == "repair":
        return PolicyDecision("repair", "Output violates language policy.")
    return PolicyDecision("fail", "Output violates language policy.")
