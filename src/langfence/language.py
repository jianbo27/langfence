from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from importlib import import_module
from importlib.util import find_spec
from threading import Lock
from typing import Any, Literal

LanguageAction = Literal["warn", "fail", "retry", "repair"]
LanguageDetector = Literal["heuristic", "lingua", "auto"]

LANGUAGE_ACTIONS: tuple[str, ...] = ("warn", "fail", "retry", "repair")
LANGUAGE_DETECTORS: tuple[str, ...] = ("heuristic", "lingua", "auto")

_LINGUA_MISSING = object()
_LINGUA_LOCK = Lock()
_lingua_detector_instance: Any = _LINGUA_MISSING


def lingua_available() -> bool:
    return find_spec("lingua") is not None


@dataclass(frozen=True)
class LanguagePolicy:
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    action: LanguageAction = "fail"
    min_confidence: float = 0.75
    detector: str = "heuristic"
    exclude_threshold: float = 0.20

    def __init__(
        self,
        include: list[str] | tuple[str, ...] = (),
        exclude: list[str] | tuple[str, ...] = (),
        action: LanguageAction = "fail",
        min_confidence: float = 0.75,
        detector: str = "heuristic",
        exclude_threshold: float = 0.20,
    ) -> None:
        if action not in LANGUAGE_ACTIONS:
            raise ValueError(
                f"Unsupported language action: {action!r} (expected one of {LANGUAGE_ACTIONS})"
            )
        if detector not in LANGUAGE_DETECTORS:
            raise ValueError(
                f"Unsupported language detector: {detector!r} "
                f"(expected one of {LANGUAGE_DETECTORS})"
            )
        if detector == "lingua" and not lingua_available():
            raise ValueError(
                "Language detector 'lingua' requires the optional dependency "
                "lingua-language-detector; install it with: pip install 'langfence[language]'. "
                "Use detector='auto' to fall back to the built-in heuristic when lingua is "
                "unavailable."
            )
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if not 0.0 <= exclude_threshold <= 1.0:
            raise ValueError("exclude_threshold must be between 0.0 and 1.0")
        object.__setattr__(self, "include", tuple(code.strip().lower() for code in include))
        object.__setattr__(self, "exclude", tuple(code.strip().lower() for code in exclude))
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "min_confidence", min_confidence)
        object.__setattr__(self, "detector", detector)
        object.__setattr__(self, "exclude_threshold", exclude_threshold)


@dataclass(frozen=True)
class LanguageDetection:
    language: str
    confidence: float
    detector: str
    metadata: dict[str, float] = field(default_factory=dict)


def detect_language(text: str, detector: str = "heuristic") -> LanguageDetection:
    if detector not in LANGUAGE_DETECTORS:
        raise ValueError(
            f"Unsupported language detector: {detector!r} (expected one of {LANGUAGE_DETECTORS})"
        )

    if detector == "lingua":
        lingua_result = _detect_with_lingua(text)
        if lingua_result is None:
            raise RuntimeError(
                "Language detector 'lingua' was requested but lingua-language-detector is not "
                "installed; install it with: pip install 'langfence[language]'"
            )
        return lingua_result

    if detector == "auto":
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

    scores: dict[str, float] = {}
    for value in values:
        score = float(value.value)
        if score > 0.0:
            scores[value.language.iso_code_639_1.name.lower()] = score

    best = values[0]
    return LanguageDetection(
        language=best.language.iso_code_639_1.name.lower(),
        confidence=float(best.value),
        detector="lingua",
        metadata=scores,
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
                # Models load lazily per language; eager preloading would hold every
                # language model in memory at once.
                _lingua_detector_instance = (
                    lingua.LanguageDetectorBuilder.from_all_languages().build()
                )

    return _lingua_detector_instance


def _is_han(codepoint: int) -> bool:
    return (
        0x4E00 <= codepoint <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= codepoint <= 0x4DBF  # Extension A
        or 0xF900 <= codepoint <= 0xFAFF  # Compatibility Ideographs
        or 0x20000 <= codepoint <= 0x3FFFF  # Extensions B and beyond
    )


def _is_kana(codepoint: int) -> bool:
    return (
        0x3040 <= codepoint <= 0x30FF  # Hiragana + Katakana
        or 0x31F0 <= codepoint <= 0x31FF  # Katakana Phonetic Extensions
        or 0xFF66 <= codepoint <= 0xFF9F  # Halfwidth Katakana
    )


def _is_hangul(codepoint: int) -> bool:
    return (
        0xAC00 <= codepoint <= 0xD7AF  # Hangul Syllables
        or 0x1100 <= codepoint <= 0x11FF  # Hangul Jamo
        or 0x3130 <= codepoint <= 0x318F  # Hangul Compatibility Jamo
        or 0xA960 <= codepoint <= 0xA97F  # Hangul Jamo Extended-A
        or 0xD7B0 <= codepoint <= 0xD7FF  # Hangul Jamo Extended-B
    )


def _detect_with_heuristics(text: str) -> LanguageDetection:
    han = 0
    kana = 0
    hangul = 0
    latin = 0
    total = 0

    for char in text:
        # Count only letters: digits, punctuation, and symbols carry no language
        # signal and would dilute the scores of short CJK sentences.
        if not unicodedata.category(char).startswith("L"):
            continue

        total += 1
        codepoint = ord(char)
        if _is_han(codepoint):
            han += 1
        elif _is_kana(codepoint):
            kana += 1
        elif _is_hangul(codepoint):
            hangul += 1
        elif codepoint <= 0x024F:  # Basic Latin through Latin Extended-B
            latin += 1

    if total == 0:
        return LanguageDetection(language="unknown", confidence=0.0, detector="heuristic")

    scores = {
        "zh": han / total,
        "ja": (kana + han * 0.35) / total,
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
