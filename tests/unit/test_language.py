from __future__ import annotations

import pytest

from langfence.language import (
    LanguageDetection,
    LanguagePolicy,
    detect_language,
    language_instruction,
)


def test_heuristic_chinese_excludes_punctuation_from_total() -> None:
    detection = detect_language("你好！")

    assert detection.language == "zh"
    assert detection.confidence == 1.0
    assert detection.detector == "heuristic"


def test_heuristic_halfwidth_katakana_is_japanese() -> None:
    detection = detect_language("ｶﾀｶﾅ")

    assert detection.language == "ja"
    assert detection.confidence == 1.0


def test_heuristic_hangul_syllables_is_korean() -> None:
    detection = detect_language("안녕하세요")

    assert detection.language == "ko"
    assert detection.confidence == 1.0


def test_heuristic_hangul_jamo_is_korean() -> None:
    # U+1100 (HANGUL CHOSEONG KIYEOK) + U+1161 (HANGUL JUNGSEONG A).
    detection = detect_language("가")

    assert detection.language == "ko"


def test_heuristic_supplementary_han_ideograph_counts_as_han() -> None:
    # U+20B9F 𠮟 lives in CJK Extension B and must be classified as han.
    detection = detect_language("\U00020b9f")

    assert detection.language == "zh"
    assert detection.confidence == 1.0


def test_heuristic_cjk_compatibility_ideograph_counts_as_han() -> None:
    # U+F900 豈 is a CJK Compatibility Ideograph.
    detection = detect_language("豈")

    assert detection.language == "zh"


def test_heuristic_digits_and_punctuation_only_is_unknown() -> None:
    detection = detect_language("42.")

    assert detection.language == "unknown"
    assert detection.confidence == 0.0
    assert detection.metadata == {}


def test_heuristic_empty_string_is_unknown() -> None:
    detection = detect_language("")

    assert detection.language == "unknown"
    assert detection.confidence == 0.0


def test_heuristic_mixed_chinese_and_english_scores_are_proportional() -> None:
    detection = detect_language("你好 hello")

    # Two han letters, five latin letters, seven letters total.
    assert detection.metadata["zh"] == pytest.approx(2 / 7)
    assert detection.metadata["en"] == pytest.approx(5 / 7)
    assert detection.language == "en"


def test_heuristic_predominantly_chinese_with_english_tail_stays_chinese() -> None:
    detection = detect_language("这是一个中文回答只有一个e")

    assert detection.language == "zh"
    assert detection.metadata["zh"] > detection.metadata["en"]


def test_heuristic_minority_han_among_latin_reports_latin() -> None:
    # A single han letter among ten latin letters: en dominates (10/11) and the
    # zh score (1/11) is still above the 0.10 floor, so a language is reported.
    detection = detect_language("abcdefghij字")

    assert detection.metadata["zh"] == pytest.approx(1 / 11)
    assert detection.language == "en"


def test_heuristic_top_score_below_floor_is_unknown() -> None:
    # Cyrillic letters count toward `total` (they are Unicode letters) but match
    # no bucket, so a single han letter among twenty Cyrillic letters scores
    # 1/21 < 0.10 for every language -> unknown.
    text = "字" + "".join(chr(0x0410 + i) for i in range(20))
    detection = detect_language(text)

    assert detection.metadata["zh"] == pytest.approx(1 / 21)
    assert max(detection.metadata.values()) < 0.10
    assert detection.language == "unknown"


@pytest.mark.parametrize("action", ["warn", "fail", "retry", "repair"])
def test_policy_accepts_every_valid_action(action: str) -> None:
    policy = LanguagePolicy(action=action)  # type: ignore[arg-type]

    assert policy.action == action


def test_policy_rejects_unknown_action() -> None:
    with pytest.raises(ValueError, match="Unsupported language action"):
        LanguagePolicy(action="bogus")  # type: ignore[arg-type]


def test_policy_rejects_unknown_detector() -> None:
    with pytest.raises(ValueError, match="Unsupported language detector"):
        LanguagePolicy(detector="bogus")


def test_policy_rejects_min_confidence_above_one() -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        LanguagePolicy(min_confidence=1.5)


def test_policy_rejects_min_confidence_below_zero() -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        LanguagePolicy(min_confidence=-0.1)


def test_policy_rejects_exclude_threshold_above_one() -> None:
    with pytest.raises(ValueError, match="exclude_threshold"):
        LanguagePolicy(exclude_threshold=1.01)


def test_policy_rejects_exclude_threshold_below_zero() -> None:
    with pytest.raises(ValueError, match="exclude_threshold"):
        LanguagePolicy(exclude_threshold=-0.5)


def test_policy_accepts_boundary_confidence_values() -> None:
    policy = LanguagePolicy(min_confidence=0.0, exclude_threshold=1.0)

    assert policy.min_confidence == 0.0
    assert policy.exclude_threshold == 1.0


def test_policy_normalizes_language_codes_to_lowercase_and_strips() -> None:
    policy = LanguagePolicy(include=["ZH ", " En"], exclude=["  JA"])

    assert policy.include == ("zh", "en")
    assert policy.exclude == ("ja",)


def test_policy_accepts_tuple_and_list_codes_equally() -> None:
    from_list = LanguagePolicy(include=["zh"])
    from_tuple = LanguagePolicy(include=("zh",))

    assert from_list.include == from_tuple.include == ("zh",)


def test_detect_language_rejects_unknown_detector() -> None:
    with pytest.raises(ValueError, match="Unsupported language detector"):
        detect_language("hello", detector="bogus")


def test_language_instruction_includes_and_excludes() -> None:
    policy = LanguagePolicy(include=["zh"], exclude=["en"])

    instruction = language_instruction(policy)

    assert "Use only these natural languages: zh." in instruction
    assert "Do not use these natural languages: en." in instruction


def test_language_instruction_empty_when_no_policy_codes() -> None:
    assert language_instruction(LanguagePolicy()) == ""


def test_lingua_unavailable_makes_policy_construction_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("langfence.language.lingua_available", lambda: False)

    with pytest.raises(ValueError, match="lingua"):
        LanguagePolicy(detector="lingua")


def test_detect_language_lingua_unavailable_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force the cached detector to resolve to "no lingua" so the lingua path
    # reports the missing optional dependency instead of loading real models.
    monkeypatch.setattr("langfence.language._lingua_detector_instance", None)

    with pytest.raises(RuntimeError, match="lingua"):
        detect_language("hello", detector="lingua")


def test_detect_language_auto_falls_back_to_heuristic_without_lingua(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("langfence.language._lingua_detector_instance", None)

    detection = detect_language("这是一个中文回答。", detector="auto")

    assert detection.detector == "heuristic"
    assert detection.language == "zh"


def test_detection_metadata_defaults_to_empty_dict() -> None:
    detection = LanguageDetection(language="zh", confidence=1.0, detector="heuristic")

    assert detection.metadata == {}


def test_lingua_policy_construction_succeeds_when_installed() -> None:
    pytest.importorskip("lingua")

    policy = LanguagePolicy(detector="lingua")

    assert policy.detector == "lingua"


def test_lingua_detects_chinese_with_metadata_scores() -> None:
    pytest.importorskip("lingua")

    detection = detect_language("这是一个中文回答，用于测试语言检测。", detector="lingua")

    assert detection.detector == "lingua"
    assert detection.language == "zh"
    assert detection.metadata
    assert all(0.0 < score <= 1.0 for score in detection.metadata.values())


def test_lingua_detects_english() -> None:
    pytest.importorskip("lingua")

    detection = detect_language(
        "This is a fairly long English sentence used to test language detection.",
        detector="lingua",
    )

    assert detection.detector == "lingua"
    assert detection.language == "en"
