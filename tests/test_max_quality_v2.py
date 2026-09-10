from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from callforge.asr import ASRSpec, candidate_defects, choose_candidate, parse_spec
from callforge.asr_benchmark import _promotion
from callforge.audio_files import audio_mime_type, colliding_transcript_targets
from callforge.config import AppConfig, GlossaryTerm
from callforge.db import Database
from callforge.evaluation import sample_calls
from callforge.lexicon import (alignment_requests, apply_alignment,
                               guard_unverified_entities, normalize_result)
from callforge.quality import build_evidence, consensus_for_row, validate_timeline
from callforge.scanner import discover_audio, scan


def test_zero_duration_decoder_loop_is_one_failed_decode_unit():
    words = [{"start": 0.0, "end": 0.0, "word": "۶"} for _ in range(15)]
    enhanced = {"segments": [{"start": 0.0, "end": 29.98, "text": "۶" * 15,
                               "words": words, "compression_ratio": 3.1}]}
    rows = build_evidence({"segments": []}, enhanced, 30.0)["segments"]
    assert len(rows) == 1
    assert (rows[0]["start"], rows[0]["end"]) == (0.0, 29.98)
    assert "failed_decode" in rows[0]["flags"]
    assert len(rows[0]["alternative_sources"]) == 1


def test_broken_alternative_does_not_poison_valid_primary_decode():
    raw = {"segments": [{"start": 0, "end": 3, "text": "سلام وقت بخیر",
                          "words": [{"start": 0, "end": 1, "word": "سلام ", "probability": .9},
                                    {"start": 1, "end": 2, "word": "وقت ", "probability": .9},
                                    {"start": 2, "end": 3, "word": "بخیر", "probability": .9}]}]}
    broken = {"segments": [{"start": 0, "end": 3, "text": "۶" * 40,
                              "compression_ratio": 3.0}]}
    row = build_evidence(raw, broken, 4)["segments"][0]
    assert row["consensus_text"] == "سلام وقت بخیر"
    assert "repetition" not in row["flags"]
    assert "alternative_failed_decode" in row["flags"]


def test_adjacent_compressed_primary_loop_rows_collapse_to_one_failed_decode():
    raw = {"segments": [
        {"start": 1, "end": 2, "text": "شکر می‌کنم", "compression_ratio": 8.0},
        {"start": 2, "end": 3, "text": "شکر می‌کنم", "compression_ratio": 8.0},
        {"start": 3, "end": 4, "text": "شکر می‌کنم", "compression_ratio": 8.0},
    ]}
    rows = build_evidence(raw, {"segments": []}, 5)["segments"]
    assert len(rows) == 1
    assert (rows[0]["start"], rows[0]["end"], rows[0]["text"]) == (1, 4, "[نامفهوم]")
    assert "failed_decode" in rows[0]["flags"]


@pytest.mark.parametrize("rows", [
    [{"start": 0, "end": 1}, {"start": 0, "end": 1}],
    [{"start": 0, "end": 2}, {"start": 1, "end": 3}],
    [{"start": 1, "end": 1}],
])
def test_invalid_published_timelines_are_rejected(rows):
    with pytest.raises(ValueError):
        validate_timeline(rows, 4)


def test_consensus_preserves_agreed_context_and_quarantines_number():
    result = consensus_for_row({"text": "مبلغ ۲۰۰ تومان ثبت شد",
                                "alternative": "مبلغ ۹۰۰ تومان ثبت شد"})
    assert result["consensus_text"] == "مبلغ [نامفهوم] تومان ثبت شد"
    assert result["reason"] == "sensitive_disagreement"
    agreed = consensus_for_row({"text": "سلام وقت بخیر", "alternative": "سلام وقت بخیر"})
    assert agreed["confidence_tier"] == "high"


def test_strong_primary_word_alignment_can_resolve_ordinary_model_disagreement():
    result = consensus_for_row({
        "text": "کارشناس مربوطه با شما تماس می‌گیرد",
        "alternative": "کارشان مربوطه با اتون تماس می‌گیره",
        "words": [{"word": word, "probability": .82}
                  for word in ("کارشناس", "مربوطه", "با", "شما", "تماس", "می‌گیرد")],
    })
    assert result["consensus_text"] == "کارشناس مربوطه با شما تماس می‌گیرد"
    assert result["reason"] == "strong_primary_alignment_over_disagreement"


def test_formats_are_case_insensitive_and_stem_collisions_fail_closed(tmp_path):
    paths = []
    for index, suffix in enumerate((".MP3", ".wav", ".M4A", ".flac", ".OGG")):
        path = tmp_path / f"unique-{index}{suffix}"
        path.write_bytes(b"not real audio")
        paths.append(path)
    left, right = tmp_path / "same.mp3", tmp_path / "same.wav"
    left.write_bytes(b"x"); right.write_bytes(b"y")
    assert set(discover_audio(tmp_path)) == set(paths + [left, right])
    assert set(colliding_transcript_targets(paths + [left, right])) == {left.resolve(), right.resolve()}
    config = AppConfig.for_root(tmp_path); config.ensure()
    database = Database(config.database); database.initialize()
    result = scan(config, database)
    assert result.discovered == 7 and result.collisions == 2


def test_v2_config_migration_is_idempotent_and_new_keys_win(tmp_path):
    workspace = tmp_path / ".callforge"; workspace.mkdir()
    config_path = workspace / "config.toml"
    config_path.write_text('[callforge]\nwhisper_model = "turbo-q4"\n\n[extra]\nvalue = 1\n')
    first = AppConfig.for_root(tmp_path)
    assert first.asr_primary.endswith("large-v3-turbo-q4")
    first.ensure(); migrated = config_path.read_text()
    AppConfig.for_root(tmp_path).ensure()
    assert config_path.read_text() == migrated
    assert AppConfig.for_root(tmp_path).asr_primary == "mlx-whisper:large-v3-turbo"
    assert "[extra]" in migrated


def test_structured_glossary_parses_aliases_and_sensitive_names(tmp_path):
    workspace = tmp_path / ".callforge"; workspace.mkdir()
    (workspace / "config.toml").write_text('''
[callforge]
glossary = ["قدیمی"]

[[callforge.terms]]
canonical = "ونسی"
type = "brand"
aliases = ["وینسی", "اینسی"]

[[callforge.terms]]
canonical = "فاضلی"
type = "support_name"
aliases = ["آجلی", "آزلیه"]
contexts = ["هستم"]
min_alignment_score = 0.5
min_alignment_margin = 0.1
min_character_hits = 0.7
''', encoding="utf-8")
    config = AppConfig.for_root(tmp_path)
    assert config.terms[0].canonical == "ونسی"
    assert config.terms[0].aliases == ("وینسی", "اینسی")
    assert config.terms[1].requires_acoustic_validation
    assert config.terms[1].min_alignment_score == .5
    assert config.terms[1].min_character_hits == .7


def test_brand_alias_normalization_preserves_word_timing_and_does_not_insert_names():
    brand = GlossaryTerm("ونسی", "brand", ("وینسی", "اینسی"))
    bank = GlossaryTerm("بلوبانک", "brand", ("بلو بانک", "بولو بانک"))
    person = GlossaryTerm("فاضلی", "support_name", ("آجلی",), ("هستم",))
    result = {"segments": [{"text": "آجلی هستم از وینسی و بلو بانک",
                             "words": [
                                 {"word": "آجلی", "start": 0, "end": .4, "probability": .7},
                                 {"word": " هستم", "start": .4, "end": .8, "probability": .8},
                                 {"word": " از", "start": .8, "end": 1, "probability": .9},
                                 {"word": " وینسی", "start": 1, "end": 1.4, "probability": .6},
                                 {"word": " و", "start": 1.4, "end": 1.5, "probability": .9},
                                 {"word": " بلو", "start": 1.5, "end": 1.8, "probability": .8},
                                 {"word": " بانک", "start": 1.8, "end": 2.1, "probability": .7},
                             ]}]}
    assert normalize_result(result, (brand, bank, person)) == 2
    segment = result["segments"][0]
    assert segment["text"] == "آجلی هستم از ونسی و بلوبانک"
    assert len(segment["words"]) == 6
    assert segment["words"][-1]["start"] == 1.5 and segment["words"][-1]["end"] == 2.1


def test_support_name_requires_ctc_score_and_runner_up_margin():
    fazeli = GlossaryTerm("فاضلی", "support_name", ("آجلی", "آزلیه"), ("هستم",), .45, .08)
    mohammadi = GlossaryTerm("محمدی", "support_name", ("ممدی",), ("هستم",), .45, .08)
    evidence = {"segments": [{"id": "s1", "start": 4, "end": 8,
                               "text": "[نامفهوم]", "alternative": "آزلیه هستم از وینسی",
                               "flags": [], "words": [],
                               "alternative_sources": [{"text": " آزلیه", "start": 4.8, "end": 5.3}],
                               "retry": {}}]}
    requests, groups = alignment_requests(evidence, (fazeli, mohammadi))
    assert [row["text"] for row in requests] == ["فاضلی", "محمدی"]
    alignment = {"segments": [
        {"id": requests[0]["id"], "words": [{"accepted": True, "score": .72, "character_hits": .8}]},
        {"id": requests[1]["id"], "words": [{"accepted": True, "score": .51, "character_hits": .7}]},
    ]}
    decisions = apply_alignment(evidence, groups, alignment)
    assert decisions[0]["accepted"]
    assert evidence["segments"][0]["alternative"] == "فاضلی هستم از وینسی"
    assert "lexicon_ctc_validated" in evidence["segments"][0]["flags"]

    evidence = {"segments": [{"id": "s1", "start": 4, "end": 8,
                               "text": "[نامفهوم]", "alternative": "آزلیه هستم",
                               "flags": [], "words": [],
                               "alternative_sources": [{"text": "آزلیه", "start": 4.8, "end": 5.3}]}]}
    requests, groups = alignment_requests(evidence, (fazeli, mohammadi))
    close = {"segments": [
        {"id": requests[0]["id"], "words": [{"accepted": True, "score": .58, "character_hits": .8}]},
        {"id": requests[1]["id"], "words": [{"accepted": True, "score": .55, "character_hits": .7}]},
    ]}
    assert not apply_alignment(evidence, groups, close)[0]["accepted"]
    assert evidence["segments"][0]["alternative"] == "آزلیه هستم"
    row = evidence["segments"][0]
    row["consensus_text"] = "آزلیه هستم"
    guard_unverified_entities(row)
    assert row["consensus_text"] == "[نامفهوم] هستم"
    assert row["confidence_tier"] == "unresolved"


def test_exact_40_reference_split_is_30_10():
    rows = [{"id": index, "duration_seconds": 60, "direction": "inbound"} for index in range(40)]
    selected = sample_calls(rows, 40, 42)
    assert sum(row["evaluation_split"] == "development" for row in selected) == 30
    assert sum(row["evaluation_split"] == "holdout" for row in selected) == 10


def test_promotion_gate_requires_quality_reductions_and_three_repeats():
    snapshots = [{"baseline_text_score": {"wer": .5, "cer": .4, "number_precision": .9},
                  "baseline_unresolved_token_rate": .2} for _ in range(30)]
    entries = [{"unsupported_numbers": False, "decoder_loop": False,
                "duplicate_intervals": False} for _ in range(90)]
    winner = {"metrics": {"macro_wer": .39, "macro_cer": .31,
                           "unresolved_token_rate": .12, "number_precision": .9,
                           "runs": 90}, "entries": entries}
    assert _promotion(winner, snapshots, 10)["eligible"]
    winner["metrics"]["runs"] = 60
    assert not _promotion(winner, snapshots, 10)["eligible"]


@pytest.mark.parametrize(("name", "mime"), [
    ("a.mp3", "audio/mpeg"), ("a.wav", "audio/wav"), ("a.m4a", "audio/mp4"),
    ("a.flac", "audio/flac"), ("a.ogg", "audio/ogg"),
])
def test_real_audio_mime(name, mime):
    assert audio_mime_type(Path(name)) == mime


def test_provider_schema_and_window_quality_selection():
    assert parse_spec("turbo") == ASRSpec("mlx-whisper", "large-v3-turbo")
    bad = {"text": "۶" * 20, "segments": [{"start": 0, "end": 1, "text": "۶" * 20}]}
    good = {"text": "سلام", "segments": [{"start": 0, "end": 1, "text": "سلام", "avg_logprob": -.2}]}
    assert "repetition" in candidate_defects(bad)
    assert choose_candidate([("agc", bad), ("raw", good)])[0] == "raw"


def test_runtime_is_offline_and_turbo_alias_is_not_q4(tmp_path):
    config = AppConfig.for_root(tmp_path)
    assert config.runtime_environment()["HF_HUB_OFFLINE"] == "1"
    script = Path(__file__).parents[1] / "src/callforge/resources/pbx-call-transcriber/scripts/transcribe_audio.py"
    source = script.read_text(encoding="utf-8")
    assert '"turbo": "mlx-community/whisper-large-v3-turbo"' in source
    assert '"turbo-q4": "mlx-community/whisper-large-v3-turbo-q4"' in source


def test_stereo_independence_heuristic_detects_alternating_sides():
    np = pytest.importorskip("numpy")
    script = Path(__file__).parents[1] / "src/callforge/resources/pbx-call-transcriber/scripts/prepare_audio.py"
    spec = importlib.util.spec_from_file_location("prepare_audio_test", script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rate = 16000
    samples = np.zeros((rate * 4, 2), dtype=np.float32)
    tone = np.sin(np.arange(rate) * 2 * math.pi * 220 / rate).astype(np.float32) * .1
    samples[:rate, 0] = tone
    samples[rate * 2:rate * 3, 1] = tone
    independent, metrics = module.independent_conversation_channels(samples, rate)
    assert independent
    assert metrics["exclusive_activity"] >= .15
