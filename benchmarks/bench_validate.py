from __future__ import annotations

import time
from collections.abc import Callable

from langfence import (
    ChoiceConstraint,
    JsonSchemaConstraint,
    LanguagePolicy,
    OutputContract,
    RegexConstraint,
    validate_output,
)
from langfence.language import detect_language

_JSON_CONTRACT = OutputContract(
    format=JsonSchemaConstraint(
        schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
    )
)
_REGEX_CONTRACT = OutputContract(format=RegexConstraint(r"\d{3}-\d{4}"))
_CHOICE_CONTRACT = OutputContract(format=ChoiceConstraint(["approved", "rejected"]))
_LANGUAGE_CONTRACT = OutputContract(
    language=LanguagePolicy(include=["zh"], exclude=["en"], min_confidence=0.2)
)


def _time(label: str, fn: Callable[[], object], iterations: int) -> None:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - start
    print(
        {
            "case": label,
            "iterations": iterations,
            "seconds": round(elapsed, 4),
            "per_second": round(iterations / elapsed),
        }
    )


def main() -> None:
    iterations = 10_000

    _time(
        "json_schema_valid",
        lambda: validate_output('{"answer": "ok"}', _JSON_CONTRACT),
        iterations,
    )
    _time(
        "json_schema_invalid",
        lambda: validate_output("{ not json", _JSON_CONTRACT),
        iterations,
    )
    _time(
        "regex_valid",
        lambda: validate_output("123-4567", _REGEX_CONTRACT),
        iterations,
    )
    _time(
        "choice_valid",
        lambda: validate_output("approved", _CHOICE_CONTRACT),
        iterations,
    )
    _time(
        "language_heuristic",
        lambda: validate_output("这是一个中文回答。", _LANGUAGE_CONTRACT),
        iterations,
    )
    _time(
        "detect_language_heuristic",
        lambda: detect_language("这是一个中文回答。"),
        iterations,
    )


if __name__ == "__main__":
    main()
