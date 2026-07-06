from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from threading import Lock
from typing import Any, Literal

LanguageAction = Literal["warn", "fail", "retry", "repair"]
_LINGUA_MISSING = object()
_LINGUA_LOCK = Lock()
_lingua_detector_instance: Any = _LINGUA_MISSING


@dataclass(frozen=True)
class LanguagePolicy:
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    action: LanguageAction = "fail"
    min_confidence: float = 0.75
    detector: str = "heuristic"

    def __init__(
        self,
        include: list[str] | tuple[str, ...] = (),
        exclude: list[str] | tuple[str, ...] = (),
        action: LanguageAction = "fail",
        min_confidence: float = 0.75,
        detector: str = "heuristic",
    ) -> None:
        object.__setattr__(self, "include", tuple(include))
        object.__setattr__(self, "exclude", tuple(exclude))
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "min_confidence", min_confidence)
        object.__setattr__(self, "detector", detector)


@dataclass(frozen=True)
class LanguageDetection:
    language: str
    confidence: float
    detector: str
    metadata: dict[str, float] = field(default_factory=dict)


def detect_language(text: str, detector: str = "heuristic") -> LanguageDetection:
    if detector == "lingua":
        lingua_result = _detect_with_lingua(text)
        if lingua_result is not None:
            return lingua_result

    return _detect_with_heuristics(text)


def _detect_with_lingua(text: str) -> LanguageDetection | None:
    detector = _lingua_detector()
    if detector is None:
        return None

    values = detector.compute_language_confidence_values(text)
    if not values:
        return LanguageDetection(language="unknown", confidence=0.0, detector="lingua")

    best = values[0]
    return LanguageDetection(
        language=best.language.iso_code_639_1.name.lower(),
        confidence=float(best.value),
        detector="lingua",
    )


def _lingua_detector() -> Any | None:
    global _lingua_detector_instance
    if _lingua_detector_instance is not _LINGUA_MISSING:
        return _lingua_detector_instance

    with _LINGUA_LOCK:
        if _lingua_detector_instance is _LINGUA_MISSING:
            try:
                lingua = import_module("lingua")
            except ImportError:
                _lingua_detector_instance = None
            else:
                _lingua_detector_instance = (
                    lingua.LanguageDetectorBuilder.from_all_languages()
                    .with_preloaded_language_models()
                    .build()
                )

    return _lingua_detector_instance


def _detect_with_heuristics(text: str) -> LanguageDetection:
    cjk = 0
    kana = 0
    hangul = 0
    latin = 0
    total = 0

    for char in text:
        if char.isspace():
            continue

        total += 1
        codepoint = ord(char)
        if 0x3400 <= codepoint <= 0x9FFF:
            cjk += 1
        if 0x3040 <= codepoint <= 0x30FF:
            kana += 1
        if 0xAC00 <= codepoint <= 0xD7AF:
            hangul += 1
        if "A" <= char <= "Z" or "a" <= char <= "z":
            latin += 1

    if total == 0:
        return LanguageDetection(language="unknown", confidence=0.0, detector="heuristic")

    scores = {
        "zh": cjk / total,
        "ja": (kana + cjk * 0.35) / total,
        "ko": hangul / total,
        "en": latin / total,
    }
    language, score = max(scores.items(), key=lambda item: item[1])
    if score < 0.10:
        language = "unknown"

    return LanguageDetection(
        language=language,
        confidence=min(score, 1.0),
        detector="heuristic",
        metadata=scores,
    )


def language_instruction(policy: LanguagePolicy) -> str:
    parts: list[str] = []
    if policy.include:
        parts.append("Use only these natural languages: " + ", ".join(policy.include) + ".")
    if policy.exclude:
        parts.append("Do not use these natural languages: " + ", ".join(policy.exclude) + ".")
    if not parts:
        return ""
    return " ".join(parts)
