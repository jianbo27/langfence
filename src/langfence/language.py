from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

LanguageAction = Literal["warn", "fail", "retry", "repair"]


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


_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_HIRAGANA_KATAKANA_RE = re.compile(r"[\u3040-\u30ff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_language(text: str, detector: str = "heuristic") -> LanguageDetection:
    if detector == "lingua":
        lingua_result = _detect_with_lingua(text)
        if lingua_result is not None:
            return lingua_result

    return _detect_with_heuristics(text)


def _detect_with_lingua(text: str) -> LanguageDetection | None:
    try:
        from lingua import LanguageDetectorBuilder  # type: ignore[import-not-found]
    except ImportError:
        return None

    detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
    values = detector.compute_language_confidence_values(text)
    if not values:
        return LanguageDetection(language="unknown", confidence=0.0, detector="lingua")

    best = values[0]
    return LanguageDetection(
        language=best.language.iso_code_639_1.name.lower(),
        confidence=float(best.value),
        detector="lingua",
    )


def _detect_with_heuristics(text: str) -> LanguageDetection:
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return LanguageDetection(language="unknown", confidence=0.0, detector="heuristic")

    cjk = len(_CJK_RE.findall(text))
    kana = len(_HIRAGANA_KATAKANA_RE.findall(text))
    hangul = len(_HANGUL_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    total = max(len(chars), 1)

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
