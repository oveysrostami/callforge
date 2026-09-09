import importlib.util
import json
import math
from pathlib import Path

import pytest

from callforge.json_utils import dumps
from callforge.quality import compact_review_input, text_flags, validate_review, write_json


def strict_load(value):
    def reject(token):
        raise ValueError(token)
    return json.loads(value, parse_constant=reject)


def helper():
    pytest.importorskip("numpy")
    path = Path(__file__).parents[1] / "src/callforge/resources/pbx-call-transcriber/scripts/transcribe_audio.py"
    spec = importlib.util.spec_from_file_location("asr_json_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nested_nonfinite_metrics_are_null_without_changing_text(tmp_path):
    data = {"text": "NaN Infinity ۱۲۳", "segments": [{"avg_logprob": float("nan"),
            "words": [{"probability": float("inf")}, {"probability": -float("inf")}, {"probability": .7}]}]}
    expected = {"text": data["text"], "segments": [{"avg_logprob": None,
                "words": [{"probability": None}, {"probability": None}, {"probability": .7}]}]}
    assert strict_load(dumps(data)) == expected
    write_json(tmp_path / "evidence.json", data)
    assert strict_load((tmp_path / "evidence.json").read_text()) == expected
    assert math.isnan(data["segments"][0]["avg_logprob"])


def test_invalid_metric_is_not_treated_as_a_valid_retry_or_number_source():
    row = {"id": "s1", "start": 0, "end": 1, "text": "سلام", "alternative": "سلام",
           "flags": ["invalid_asr_metrics"], "speaker": "گوینده نامشخص",
           "retry": {"text": "۲۲ میلیون", "segments": [{"text": "۲۲ میلیون", "avg_logprob": float("nan")}]}}
    evidence = {"duration_seconds": 1, "segments": [row]}
    assert "invalid_asr_metrics" in text_flags("سلام", {"avg_logprob": float("nan")})
    assert compact_review_input(evidence)["segments"][0]["retry_text"] is None
    reviewed = validate_review({"segments": [dict(id="s1", text="۲۲ میلیون", speaker="مشتری", uncertain=False, notes="")]}, evidence)
    assert reviewed[0]["uncertain"] is True
    assert "unsupported_number" in reviewed[0]["flags"]


def test_helper_flags_invalid_scores_and_emits_standard_json():
    module = helper()
    rows = module.sanitize_segments([{"text": "سلام", "avg_logprob": float("nan"),
                                      "words": [{"probability": float("inf")}]}])
    assert rows[0]["avg_logprob"] is None
    assert rows[0]["flags"] == ["invalid_asr_metrics"]
    assert rows[0]["invalid_metrics"] == ["avg_logprob"]
    assert strict_load(json.dumps(rows, allow_nan=False)) == rows


def test_mlx_all_masked_path_ends_with_eot_and_requests_existing_fallback():
    mx = pytest.importorskip("mlx.core")
    import mlx_whisper.decoding as decoding
    module = helper()
    original = decoding.GreedyDecoder
    with module.guarded_mlx_decoder() as warnings:
        decoder = decoding.GreedyDecoder(temperature=0, eot=2)
        tokens, complete, scores = decoder.update(mx.array([[0]]), mx.full((1, 3), -mx.inf), mx.array([-.5]))
        assert tokens.tolist() == [[0, 2]]
        assert bool(complete) and scores.tolist() == [-math.inf]
        assert scores.item() < -1.  # Existing Whisper fallback threshold.
        assert warnings == ["decoder_truncated"]
        _, _, normal = decoder.update(mx.array([[0]]), mx.array([[1., 2., 3.]]), mx.array([0.]))
        _, _, expected = original(0, 2).update(mx.array([[0]]), mx.array([[1., 2., 3.]]), mx.array([0.]))
        assert normal.tolist() == expected.tolist()
    assert decoding.GreedyDecoder is original
