import copy
import json

import pytest

np = pytest.importorskip("numpy")

from callforge.alignment import align_emissions, ctc_path, text_units
from callforge.alignment_experiment import refine_segments, run_alignment_experiment
from callforge.roles import align_segments


def emissions(labels, classes=4):
    values = np.full((len(labels), classes), -9.)
    for index, label in enumerate(labels):
        values[index, label] = -.01
    return values


def test_ctc_repeats_require_blank_and_ignore_padding():
    data = emissions([3, 0, 1, 1, 0, 1, 0, 2, 3])
    assert ctc_path(data, [1, 1, 2], 0) == [[2, 3], [5], [7]]
    with pytest.raises(ValueError, match="complete"):
        ctc_path(emissions([1, 1]), [1, 1], 0)


def test_alignment_preserves_punctuation_and_does_not_expand_numbers():
    text = "بله، ۳۲ [نامفهوم]"
    units = text_units(text, {"ب": 1, "ل": 2, "ه": 3})
    assert units[0]["token_ids"] == [1, 2, 3]
    assert all(not u["token_ids"] for u in units[1:])
    assert [text[u["char_start"]:u["char_end"]] for u in units] == ["بله،", "۳۲", "[نامفهوم]"]
    data = emissions([0, 1, 2, 3, 0, 1, 0, 2, 0])
    words = align_emissions(text, {"ب": 1, "ل": 2, "ه": 3}, 0, data, 10., .02)
    assert words[0]["accepted"] and words[0]["start"] == pytest.approx(10.02)
    assert all(w["start"] is None and not w["accepted"] for w in words[1:])


def fixture():
    rows = [{"id": "s1", "text": "سلام خوبی", "start": 0., "end": 2.}]
    turns = [{"start": 0., "end": .8, "speaker_id": "A"}, {"start": 1., "end": 2., "speaker_id": "B"}]
    words = [{"text": "سلام", "char_start": 0, "char_end": 4, "start": 1.1, "end": 1.3,
              "accepted": True, "score": .9, "character_hits": 1.},
             {"text": "خوبی", "char_start": 5, "char_end": 9, "start": 1.5, "end": 1.8,
              "accepted": True, "score": .9, "character_hits": 1.}]
    return align_segments(rows, turns), turns, {"segments": [{"id": "s1", "status": "completed", "words": words}]}


def test_word_local_refinement_can_resolve_stale_boundaries_without_rewriting():
    rows, turns, alignment = fixture()
    before = copy.deepcopy(rows)
    result = refine_segments(rows, alignment, turns)
    assert rows == before
    assert result[0]["speaker_id"] == "B"
    assert result[0]["uncertain"] and "boundary_review_required" in result[0]["flags"]
    assert [(r["start"], r["end"], r["text"]) for r in result] == [(0., 2., "سلام خوبی")]


def test_mixed_words_do_not_get_one_majority_role():
    rows, turns, alignment = fixture()
    alignment["segments"][0]["words"][0].update(start=.1, end=.5)
    result = refine_segments(rows, alignment, turns)
    assert result[0]["speaker_id"] is None
    assert "mixed_speakers" in result[0]["flags"]


@pytest.mark.parametrize("change", ["missing", "invented", "order", "out_of_bounds"])
def test_refinement_rejects_incomplete_or_changed_alignment(change):
    rows, turns, alignment = fixture()
    words = alignment["segments"][0]["words"]
    if change == "missing": words.pop()
    if change == "invented": words[0]["text"] = "new"
    if change == "order": words.reverse()
    if change == "out_of_bounds": words[0]["end"] = 20
    with pytest.raises(ValueError):
        refine_segments(rows, alignment, turns)


def test_overlap_low_score_and_missing_alignment_stay_unknown():
    rows, turns, alignment = fixture()
    assert refine_segments(rows, {"segments": []}, turns)[0]["speaker_id"] is None
    for word in alignment["segments"][0]["words"]:
        word["accepted"] = False
    assert refine_segments(rows, alignment, turns)[0]["speaker_id"] is None
    for word in alignment["segments"][0]["words"]:
        word["accepted"] = True
    overlap = turns + [dict(turns[1], speaker_id="C")]
    assert refine_segments(rows, alignment, overlap)[0]["speaker_id"] is None


def test_attention_alignment_keeps_literal_offsets_and_refuses_rewritten_tokens():
    from types import SimpleNamespace
    from callforge.alignment_mlx_worker import literal_words
    pieces = [SimpleNamespace(word=" سلام", start=.1, end=.3, probability=.9),
              SimpleNamespace(word="،", start=.3, end=.3, probability=.9),
              SimpleNamespace(word=" خوبی", start=.4, end=.8, probability=.8)]
    words = literal_words("سلام، خوبی", pieces, 10.)
    assert [(w["text"], w["char_start"], w["char_end"]) for w in words] == [("سلام،", 0, 5), ("خوبی", 6, 10)]
    assert words[0]["start"] == 10.1 and words[0]["character_hits"] is None
    assert words[0]["method"] == "whisper_attention"
    with pytest.raises(ValueError, match="changed"):
        literal_words("متن متفاوت", pieces, 0.)


@pytest.mark.parametrize("fail", [False, True])
def test_alignment_experiment_no_database_writes_no_reference_leakage_and_cleanup(tmp_path, monkeypatch, fail):
    from callforge.config import AppConfig
    from callforge.quality import file_hash, write_json
    from callforge.benchmark import speaker_python
    config = AppConfig.for_root(tmp_path)
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"audio")
    audio.with_suffix(".md").write_text("approved")
    benchmark = tmp_path / "benchmark.json"
    write_json(benchmark, {"schema_version": 1, "split": "development", "audio_path": str(audio),
        "audio_sha256": file_hash(audio), "baseline": {"segments": [{"id": "s1", "start": 0, "end": 1, "text": "سلام"}]},
        "reference": {"segments": [{"text": "secret reference"}]}})
    acoustic = tmp_path / "acoustic"
    acoustic.mkdir()
    write_json(acoustic / "report.json", {"status": "completed", "source_unchanged": True,
        "audio_sha256": file_hash(audio), "benchmark_sha256": file_hash(benchmark), "runtime": {"turns": []}})
    interpreter = speaker_python(config)
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    output = tmp_path / "experiment"
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if "callforge.alignment_worker" in command:
            assert "secret reference" not in (output / "input.json").read_text()
            if fail:
                import subprocess
                raise subprocess.TimeoutExpired(command, 5)
            from callforge.alignment import MODEL, REVISION
            json.dump({"model": MODEL, "revision": REVISION,
                       "segments": [{"id": "s1", "status": "unalignable", "words": []}]}, kwargs["stdout"])
    monkeypatch.setattr("callforge.alignment_experiment.subprocess.run", run)
    if fail:
        with pytest.raises(RuntimeError): run_alignment_experiment(config, benchmark, acoustic, output)
    else:
        report = run_alignment_experiment(config, benchmark, acoustic, output)
        assert report["status"] == "completed" and report["markdown_unchanged"]
    assert len(calls) == 2
    assert audio.with_suffix(".md").read_text() == "approved"
    assert not config.database.exists() and not list(output.glob("audio-*"))
