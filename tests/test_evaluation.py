from callforge.evaluation import sample_calls, score


def test_known_wer_cer_and_number_error():
    result = score("سلام مبلغ ۲۰۰ تومان", "سلام مبلغ ۳۰۰ تومان")
    assert result["wer"] == .25
    assert result["word_errors"] == 1
    assert result["number_recall"] == 0
    assert result["number_precision"] == 0
    assert score("", "hello")["wer"] is None
    assert score("علي ۱۲۳", "علی 123")["wer"] == 0


def test_samples_are_deterministic_stratified_and_have_holdout():
    rows = [{"id": i, "duration_seconds": duration, "direction": direction}
            for i, (duration, direction) in enumerate([(20, "inbound"), (90, "inbound"), (200, "outbound")] * 10)]
    selected = sample_calls(rows, 12, 42)
    assert selected == sample_calls(list(reversed(rows)), 12, 42)
    assert len({row["id"] for row in selected}) == 12
    assert len([row for row in selected if row["evaluation_split"] == "holdout"]) == 3
    assert {row["duration_seconds"] for row in selected} == {20, 90, 200}
