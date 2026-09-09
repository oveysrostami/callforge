import copy
import importlib.util
from pathlib import Path

import pytest

from callforge.quality import (
    aligned_retry_for_gap,
    apply_speaker_evidence,
    attach_coverage_retry,
    build_evidence,
    coverage_retry_candidates,
    coverage_retry_windows,
    normalize,
    prompt_leakage,
    render_markdown,
    text_flags,
    timed_units,
    validate_review,
)


def evidence():
    raw = {"segments": [{"start": 0, "end": 2, "text": "سلام"},
                        {"start": 3, "end": 5, "text": "مبلغ ۲۰۰ تومان"}]}
    return build_evidence(raw, copy.deepcopy(raw), 6)


def response():
    return {"segments": [{"id": "s1", "text": "سلام", "speaker": "گوینده نامشخص", "uncertain": False, "notes": ""},
                         {"id": "s2", "text": "مبلغ ۲۰۰ تومان", "speaker": "مشتری", "uncertain": False, "notes": ""}]}


def test_normalization_and_repetition_are_not_confidence():
    assert normalize("علي ۱۲۳٤") == "علی 1234"
    assert "repetition" in text_flags("میفهموییییییییی")
    assert "repetition" in text_flags("از از از از از ")
    assert "verify_numbers" in text_flags("۲۰۰ تومان")
    assert prompt_leakage("این یک مکالمه تلفنی فارسی میانی است.",
                          "این یک مکالمه تلفنی فارسی میان مشتری و کارشناس پشتیبانی است.")
    assert not prompt_leakage("سلام، درخواست برداشت وجه داشتم.",
                              "این یک مکالمه تلفنی فارسی میان مشتری و کارشناس پشتیبانی است.")


@pytest.mark.parametrize("mutation", ["greeting_only", "duplicate", "unknown", "placeholder", "no_array"])
def test_review_rejects_partial_and_invalid_final_responses(mutation):
    value = response()
    if mutation == "greeting_only": value["segments"].pop()
    if mutation == "duplicate": value["segments"][1]["id"] = "s1"
    if mutation == "unknown": value["segments"][1]["id"] = "s404"
    if mutation == "placeholder": value["segments"][1]["text"] = "..."
    if mutation == "no_array": value = {"text": "سلام"}
    with pytest.raises(ValueError):
        validate_review(value, evidence())


def test_review_retains_numbers_flags_and_canonical_time():
    value = response()
    value["segments"][1]["text"] = "مبلغ ۹۰۰ تومان"
    value["segments"][1]["start"] = 99  # Untrusted response must not replace evidence timing.
    rows = validate_review(value, evidence())
    assert rows[1]["start"] == 3
    assert "unsupported_number" in rows[1]["flags"]
    assert rows[1]["uncertain"] is True
    assert "۹۰۰" not in rows[1]["text"] and "[نامفهوم]" in rows[1]["text"]
    assert rows[1]["raw_text"] == "مبلغ ۲۰۰ تومان"
    assert "نیازمند بازبینی انسانی" in render_markdown("test.mp3", rows)


def test_enhanced_and_vad_only_gaps_are_retained():
    result = build_evidence({"segments": [{"start": 0, "end": 2, "text": "سلام"}]},
                            {"segments": [{"start": 3, "end": 5, "text": "خداحافظ"}]},
                            8, [{"start": 6, "end": 8}])
    assert [(s["start"], s["end"]) for s in result["segments"]] == [(0, 2), (3, 5), (6.4, 7.6)]
    assert result["segments"][1]["alternative"] == "خداحافظ"
    assert result["segments"][2]["flags"] == ["speech_gap"]


def test_vad_padding_does_not_create_false_unknown_segments():
    raw = {"segments": [{"start": 1, "end": 2, "text": "سلام"}]}
    result = build_evidence(raw, raw, 3, [{"start": .6, "end": 2.4}])
    assert len(result["segments"]) == 1


def test_enhanced_words_have_one_owner_without_copying_neighbor_phrases():
    raw = {"segments": [{"start": 0, "end": 2, "text": "سلام"},
                        {"start": 2, "end": 4, "text": "خانم یوسفی"}]}
    enhanced = {"segments": [{"start": 0, "end": 4.5, "text": "سلام خانم یوسفی",
        "words": [{"start": 0, "end": 1, "word": "سلام"},
                  {"start": 2, "end": 2, "word": " خانم"},
                  {"start": 2.1, "end": 4.5, "word": " یوسفی"}]}]}
    before = copy.deepcopy((raw, enhanced))
    rows = build_evidence(raw, enhanced, 5)["segments"]
    assert [row["alternative"] for row in rows] == ["سلام", "خانم یوسفی"]
    assert sum(len(row["alternative_sources"]) for row in rows) == 3
    assert (raw, enhanced) == before


def test_unsplittable_enhanced_phrase_creates_joint_unit_not_copied_slivers():
    raw = {"segments": [{"start": 1, "end": 2, "text": "سلام"},
                        {"start": 3, "end": 4, "text": "خوب هستین"}]}
    enhanced = {"segments": [{"start": .5, "end": 4.5, "text": "سلام حالتون خوبه"}]}
    rows = build_evidence(raw, enhanced, 5)["segments"]
    assert len(rows) == 1
    assert rows[0]["text"] == "سلام خوب هستین"
    assert rows[0]["alternative"] == "سلام حالتون خوبه"
    assert "alternative_timing_uncertain" in rows[0]["flags"]


@pytest.mark.parametrize("bad_words", [
    [{"word": "سلام", "start": None, "end": 1}],
    [{"word": "سلام", "start": 0, "end": float("nan")}],
    [{"word": "اشتباه", "start": 0, "end": 1}],
])
def test_bad_word_alignment_preserves_whole_phrase(bad_words):
    segment = {"start": 0, "end": 2, "text": "سلام", "words": bad_words}
    rows = build_evidence({"segments": []}, {"segments": [segment]}, 3)["segments"]
    assert len(rows) == 1
    assert rows[0]["alternative"] == "سلام"
    assert rows[0]["alternative_sources"][0]["word_index"] is None


def test_real_repeated_words_and_brief_enhanced_only_reply_are_not_deduplicated():
    enhanced = {"segments": [{"start": 0, "end": 3, "text": "بله بله بله", "words": [
        {"start": 0, "end": .5, "word": "بله"},
        {"start": 1, "end": 1.1, "word": " بله"},
        {"start": 2, "end": 3, "word": " بله"}]}]}
    raw = {"segments": [{"start": 0, "end": .5, "text": "بله"},
                        {"start": 2, "end": 3, "text": "بله"}]}
    rows = build_evidence(raw, enhanced, 4)["segments"]
    assert len(rows) == 3
    assert [row["alternative"] for row in rows] == ["بله"] * 3
    assert rows[1]["start"] == 1
    assert rows[1]["end"] == 1.1


def test_enhanced_only_point_word_uses_enclosing_interval():
    enhanced = {"segments": [{"start": 1, "end": 2, "text": "بله", "words": [
        {"start": 1.5, "end": 1.5, "word": "بله"}]}]}
    rows = build_evidence({"segments": []}, enhanced, 3)["segments"]
    assert (rows[0]["start"], rows[0]["end"], rows[0]["alternative"]) == (1, 2, "بله")


def test_empty_zero_duration_decoder_segment_is_not_canonical_evidence():
    raw = {"segments": [{"start": 1, "end": 1, "text": "", "words": []}]}
    assert build_evidence(raw, {"segments": []}, 2)["segments"] == []


def test_adjacent_point_words_do_not_expand_into_duplicate_full_segments():
    enhanced = {"segments": [{"start": 1, "end": 5, "text": "سلام است. خداحافظ", "words": [
        {"start": 1, "end": 2, "word": "سلام"},
        {"start": 2, "end": 2, "word": " است."},
        {"start": 3, "end": 5, "word": " خداحافظ"}]}]}
    rows = build_evidence({"segments": []}, enhanced, 6)["segments"]
    assert [(row["start"], row["end"], row["alternative"]) for row in rows] == [
        (1, 2, "سلام است."), (3, 5, "خداحافظ")]


def test_decoder_envelope_does_not_hide_vad_speech_without_words():
    enhanced = {"segments": [{"start": 0, "end": 10, "text": "سلام", "words": [
        {"start": 0, "end": 1, "word": "سلام"}]}]}
    rows = build_evidence({"segments": []}, enhanced, 10, [{"start": 0, "end": 10}])["segments"]
    assert rows[0]["alternative"] == "سلام"
    assert any(row["start"] >= 1 and row["alternative"] == "[نامفهوم]" for row in rows)


def test_coverage_recovery_groups_missing_speech_and_includes_failed_decoder_rows():
    data = {
        "duration_seconds": 40,
        "speech_regions": [{"start": 0, "end": 11}, {"start": 29, "end": 33}],
        "segments": [
            {"id": "s1", "start": .4, "end": 3, "text": "[نامفهوم]", "alternative": "[نامفهوم]", "flags": ["speech_gap"]},
            {"id": "s2", "start": 3.5, "end": 5, "text": "[نامفهوم]", "alternative": "", "flags": ["speech_gap"]},
            {"id": "s3", "start": 8, "end": 10, "text": "۶" * 20, "alternative": "", "flags": ["repetition"]},
            {"id": "s4", "start": 30, "end": 32, "text": "[نامفهوم]", "alternative": "", "flags": ["speech_gap"]},
            {"id": "s5", "start": 34, "end": 36, "text": "[نامفهوم]", "alternative": "متن معتبر", "flags": ["speech_gap"]},
        ],
    }
    assert [row["id"] for row in coverage_retry_candidates(data)] == ["s1", "s2", "s3", "s4"]
    plans = coverage_retry_windows(data)
    assert [plan["segment_ids"] for plan in plans] == [["s1", "s2", "s3"], ["s4"]]
    assert all(plan["end"] - plan["start"] == 30 for plan in plans)
    assert all(0 <= plan["start"] < plan["end"] <= data["duration_seconds"] for plan in plans)


def test_contextual_retry_is_partitioned_by_word_time_and_rejects_loops():
    retry = {"start": 0, "end": 10, "segments": [{
        "start": 0, "end": 10, "text": "قبل سلام دنیا بعد", "avg_logprob": -.2,
        "no_speech_prob": .01, "compression_ratio": 1.1,
        "words": [
            {"start": 1, "end": 2, "word": "قبل "},
            {"start": 3.1, "end": 3.5, "word": "سلام "},
            {"start": 4.1, "end": 4.5, "word": "دنیا "},
            {"start": 7, "end": 8, "word": "بعد"},
        ],
    }]}
    aligned = aligned_retry_for_gap(retry, 3, 5)
    assert aligned is not None
    assert aligned["text"] == "سلام دنیا"
    assert aligned["segments"][0]["start"] == 3.1
    assert aligned["segments"][0]["end"] == 4.5

    data = {"segments": [{"id": "s1", "start": 3, "end": 5, "text": "[نامفهوم]", "flags": ["speech_gap"]}]}
    assert attach_coverage_retry(data, retry, ["s1"]) == 1
    assert data["segments"][0]["retry"]["text"] == "سلام دنیا"
    looping = {"segments": [{"start": 3, "end": 5, "text": "۶" * 30,
                              "words": [], "avg_logprob": -.1}]}
    assert aligned_retry_for_gap(looping, 3, 5) is None
    leaked = {"settings": {"prompt": "این یک مکالمه تلفنی فارسی میان مشتری و کارشناس پشتیبانی است."},
              "segments": [{"start": 3, "end": 5, "text": "این یک مکالمه تلفنی فارسی میانی است.",
                            "words": [], "avg_logprob": -.1}]}
    assert aligned_retry_for_gap(leaked, 3, 5) is None
    uncertain_number = {"segments": [{
        "start": 0, "end": 10, "text": "مبلغ ۳۰", "avg_logprob": -.2,
        "words": [{"start": 3, "end": 4, "word": " ۳۰", "probability": .2}],
    }]}
    assert aligned_retry_for_gap(uncertain_number, 3, 5) is None


def test_long_vad_only_region_is_split_into_canonical_units():
    result = build_evidence({"segments": []}, {"segments": []}, 30,
                            [{"start": 0, "end": 30}])
    gaps = result["segments"]
    assert len(gaps) == 4
    assert max(row["end"] - row["start"] for row in gaps) <= 8
    assert coverage_retry_windows(result)


def test_review_rejects_unrepaired_decoder_loops():
    for text in ("۶" * 100, "نه " * 30):
        value = response()
        value["segments"][0]["text"] = text
        with pytest.raises(ValueError, match="repetition loop"):
            validate_review(value, evidence())


def test_retry_prioritizes_late_decoder_loops_over_early_gaps():
    from callforge.quality import retry_priority, compact_review_input
    rows = [{"start": 0, "flags": ["speech_gap"]}, {"start": 90, "flags": ["repetition"]},
            {"start": 20, "flags": ["pass_disagreement"]}]
    assert [row["start"] for row in sorted(rows, key=retry_priority)] == [90, 20, 0]
    data = evidence()
    data["segments"][0]["words"] = [{"word": "سلام"}]
    data["segments"][0]["retry"] = {"text": "سلام", "words": [1, 2]}
    compact = compact_review_input(data)
    assert "words" not in compact["segments"][0]
    assert compact["segments"][0]["retry_text"] == "سلام"


def test_agc_is_identity_when_gain_is_one_and_does_not_clip():
    np = pytest.importorskip("numpy")
    path = Path(__file__).parents[1] / "src/callforge/resources/pbx-call-transcriber/scripts/prepare_audio.py"
    spec = importlib.util.spec_from_file_location("prepare_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    signal = np.array([0., .5, -.5, .95, -.95], dtype=np.float32)
    assert np.array_equal(signal, module.apply_agc(signal, 16000, .06, 18, .0025))
    quiet = np.full(16000, .01, dtype=np.float32)
    amplified = module.apply_agc(quiet, 16000, .06, 18, .0025)
    assert np.max(amplified) < .951
    assert module.rms(amplified) > module.rms(quiet)


def test_word_units_split_at_real_pauses_without_inventing_word_times():
    row = {"start": 0, "end": 5, "text": "سلام خوبی", "words": [
        {"word": "سلام", "start": 0, "end": 1}, {"word": " خوبی", "start": 3, "end": 4}]}
    units = timed_units([row])
    assert [unit["text"] for unit in units] == ["سلام", "خوبی"]
    assert [unit["start"] for unit in units] == [0, 3]
    row["text"] = "متنی که با کلمات زمان‌بندی‌شده مطابقت ندارد"
    assert timed_units([row]) == [row]


def test_diarization_splits_by_words_but_never_infers_roles():
    result = {"segments": [{"id": "s1", "start": 0, "end": 2, "text": "سلام خوبی", "flags": [],
                            "words": [{"word": "سلام", "start": 0, "end": 1},
                                      {"word": " خوبی", "start": 1, "end": 2}]}]}
    apply_speaker_evidence(result, [{"start": 0, "end": 1, "speaker_id": "SPEAKER_00"},
                                    {"start": 1, "end": 2, "speaker_id": "SPEAKER_01"}])
    assert [row["speaker"] for row in result["segments"]] == ["SPEAKER_00", "SPEAKER_01"]
    assert " ".join(row["text"] for row in result["segments"]) == "سلام خوبی"


def test_overlapping_speakers_are_marked_uncertain():
    result = evidence()
    apply_speaker_evidence(result, [{"start": 0, "end": 5, "speaker_id": "A"},
                                    {"start": 0, "end": 5, "speaker_id": "B"}])
    assert all(row["speaker_id"] is None for row in result["segments"])
    assert all("speaker_uncertain" in row["flags"] for row in result["segments"])
