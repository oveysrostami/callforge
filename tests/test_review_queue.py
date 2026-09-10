from callforge.review_queue import candidate_options, is_unresolved_segment


def test_review_candidates_are_safe_deduplicated_full_segments():
    row = {
        "text": "سلام من [نامفهوم] هستم",
        "raw_text": "سلام من وینسی هستم",
        "alternative": "سلام من ونسی هستم",
        "consensus_text": "سلام من [نامفهوم] هستم",
        "retry": {"text": "سلام من ونسی هستم", "segments": []},
        "uncertainty_spans": [{"candidates": ["ونسی", "وینسی", "[نامفهوم]"]}],
        "flags": ["unclear"],
        "uncertain": True,
    }
    assert is_unresolved_segment(row)
    values = candidate_options(row)
    assert [item["text"] for item in values] == [
        "سلام من وینسی هستم",
        "سلام من ونسی هستم",
    ]
    assert len({item["id"] for item in values}) == 2


def test_review_candidates_reject_loops_leakage_and_ambiguous_fragments():
    row = {
        "text": "[نامفهوم] و [نامفهوم]",
        "raw_text": "سلام سلام سلام سلام سلام",
        "alternative": "متن ظاهراً سالم",
        "flags": ["unclear", "alternative_prompt_leakage"],
        "uncertainty_spans": [{"candidates": ["ونسی"]}],
        "retry": {"text": "راهنمای نگارشی", "settings": {"prompt": "راهنمای نگارشی"}},
    }
    assert candidate_options(row) == []


def test_plain_attention_flag_without_unresolved_span_is_not_in_unclear_queue():
    assert not is_unresolved_segment({
        "text": "کد ملی‌تون رو می‌فرمایین؟",
        "uncertain": True,
        "flags": ["verify_numbers"],
        "confidence_tier": "low",
    })
