import copy
import json

import pytest

from callforge.roles import (UNKNOWN_ROLE, align_segments, apply_roles, role_input,
                             role_prompt, role_score, validate_roles)


def inputs():
    rows = [{"id": "s1", "start": 0, "end": 2, "text": "از پشتیبانی تماس گرفتم", "speaker": "human label"},
            {"id": "s2", "start": 2, "end": 4, "text": "حسابتون رو بررسی کردم"}]
    turns = [{"start": 0, "end": 4, "speaker_id": "A"}]
    return rows, turns


def valid():
    return {"roles": [{"speaker_id": "A", "role": "پشتیبان", "reason": "معرفی و بررسی حساب",
                        "evidence": [{"segment_id": "s1", "quote": "از پشتیبانی"},
                                     {"segment_id": "s2", "quote": "حسابتون رو بررسی کردم"}]}]}


def test_alignment_preserves_text_timing_and_does_not_leak_existing_roles():
    rows, turns = inputs()
    before = copy.deepcopy(rows)
    aligned = align_segments(rows, turns)
    data = role_input(aligned, "outbound")
    assert "human label" not in role_prompt(data)
    result = apply_roles(aligned, validate_roles(valid(), data))
    assert [r["speaker"] for r in result] == ["پشتیبان"] * 2
    assert [(r["text"], r["start"], r["end"]) for r in result] == [(r["text"], r["start"], r["end"]) for r in rows]
    assert rows == before


def test_alignment_counts_unions_not_duplicates_and_abstains_on_mixed_turns():
    rows, turns = inputs()
    assert align_segments(rows, turns * 2) == align_segments(rows, turns)
    assert align_segments(rows, [dict(turns[0], speaker_id="B"), *turns])[0]["speaker_id"] is None
    mixed = [{"start": 0, "end": 1, "speaker_id": "A"}, {"start": 1, "end": 2, "speaker_id": "B"}]
    assert align_segments(rows, mixed)[0]["speaker_id"] is None
    assert align_segments(rows, [dict(turns[0], end=.3)])[0]["speaker_id"] is None


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "invented_quote", "wrong_speaker", "one_citation", "internal", "bad_role"])
def test_roles_reject_ungrounded_or_incomplete_response(mutation):
    rows, turns = inputs()
    data = role_input(align_segments(rows, turns), "outbound")
    value = valid()
    if mutation == "missing": value["roles"] = []
    if mutation == "duplicate": value["roles"] *= 2
    if mutation == "invented_quote": value["roles"][0]["evidence"][0]["quote"] = "not in transcript"
    if mutation == "wrong_speaker": data["segments"][0]["speaker_id"] = None
    if mutation == "one_citation": value["roles"][0]["evidence"] = value["roles"][0]["evidence"][:1] * 2
    if mutation == "internal": data["direction"] = "internal"
    if mutation == "bad_role": value["roles"][0]["role"] = "guessed-name"
    with pytest.raises(ValueError):
        validate_roles(value, data)


def test_unknown_is_valid_and_fixed_role_metric_does_not_hide_swapped_roles():
    rows, turns = inputs()
    data = role_input(align_segments(rows, turns), "internal")
    value = {"roles": [{"speaker_id": "A", "role": UNKNOWN_ROLE, "reason": "شواهد کافی نیست", "evidence": []}]}
    assert validate_roles(value, data)["A"]["role"] == UNKNOWN_ROLE
    reference = [{"start": 0, "end": 2, "speaker": "مشتری"}, {"start": 2, "end": 4, "speaker": "پشتیبان"}]
    swapped = [dict(reference[0], speaker="پشتیبان"), dict(reference[1], speaker="مشتری")]
    assert role_score(reference, swapped)["agreement"] == 0
    assert role_score(reference, reference)["agreement"] == 1
    assert role_score(reference, [])["unknown_seconds"] == 4


def test_old_word_arrays_never_overwrite_reviewed_text():
    from callforge.quality import apply_speaker_evidence
    data = {"segments": [{"id": "s1", "start": 0, "end": 2, "text": "متن اصلاح‌شده", "flags": [],
                          "words": [{"start": 0, "end": 1, "word": "متن"}, {"start": 1, "end": 2, "word": " غلط"}]}]}
    apply_speaker_evidence(data, [{"start": 0, "end": 1, "speaker_id": "A"}, {"start": 1, "end": 2, "speaker_id": "B"}])
    assert len(data["segments"]) == 1
    assert data["segments"][0]["text"] == "متن اصلاح‌شده"


@pytest.mark.parametrize("failure", [None, "timeout", "tools", "bad_citation", "provenance"])
def test_role_experiment_is_isolated_and_fails_closed(tmp_path, monkeypatch, failure):
    from callforge.benchmark import run_role_experiment
    from callforge.config import AppConfig
    from callforge.quality import file_hash, write_json
    from types import SimpleNamespace
    config = AppConfig.for_root(tmp_path)
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")
    audio.with_suffix(".md").write_text("approved original")
    rows, turns = inputs()
    rows = [dict(r, speaker=UNKNOWN_ROLE) for r in rows]
    snapshot = {"schema_version": 1, "split": "development", "audio_path": str(audio),
                "audio_sha256": file_hash(audio), "direction": "outbound", "baseline": {"segments": rows},
                "reference": {"segments": [dict(r, speaker="پشتیبان") for r in rows]}}
    benchmark = tmp_path / "benchmark.json"
    write_json(benchmark, snapshot)
    acoustic = tmp_path / "acoustic"
    acoustic.mkdir()
    write_json(acoustic / "report.json", {"status": "completed", "source_unchanged": True,
        "benchmark_sha256": file_hash(benchmark), "audio_sha256": "bad" if failure == "provenance" else file_hash(audio),
        "runtime": {"turns": turns}})
    output = tmp_path / "result"
    monkeypatch.setattr("shutil.which", lambda _: "/fake/codex")
    monkeypatch.setattr("callforge.codex_runner.CodexRunner._configured_model", lambda _: "test-model")
    def start(command, **kwargs):
        prompt = kwargs["stdin"].read()
        assert "human label" not in prompt and str(benchmark) not in prompt
        value = valid()
        if failure == "bad_citation": value["roles"][0]["evidence"][0]["quote"] = "invented"
        write_json(output / "roles.json", value)
        if failure == "tools":
            kwargs["stdout"].write(json.dumps({"item": {"type": "command_execution"}}) + "\n")
        return SimpleNamespace(poll=lambda: 0)
    monkeypatch.setattr("callforge.benchmark.subprocess.Popen", start)
    monkeypatch.setattr("callforge.codex_runner.CodexRunner._wait_for_codex", lambda *args:
                        SimpleNamespace(returncode=124 if failure == "timeout" else 0, forced_reason="hard_timeout" if failure == "timeout" else None))
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            run_role_experiment(config, benchmark, acoustic, output)
        assert not (output / "preview.md").exists()
        if failure != "provenance": assert json.loads((output / "report.json").read_text())["status"] == "failed"
    else:
        report = run_role_experiment(config, benchmark, acoustic, output)
        assert report["status"] == "completed" and report["role_score"]["agreement"] == 1
        assert report["markdown_unchanged"] and report["source_unchanged"] and report["text_unchanged"]
        with pytest.raises(FileExistsError): run_role_experiment(config, benchmark, acoustic, output)
    assert not config.database.exists()
    assert audio.with_suffix(".md").read_text() == "approved original"
