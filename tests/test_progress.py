import json

from callforge.progress import describe_codex_event, read_progress_events


def test_describes_audio_preparation_and_whisper_without_exposing_commands():
    prepare = describe_codex_event(
        {
            "type": "item.started",
            "item": {
                "type": "command_execution",
                "status": "in_progress",
                "command": "/tmp/venv/bin/python scripts/prepare_audio.py secret.mp3",
            },
        }
    )
    whisper = describe_codex_event(
        {
            "type": "item.started",
            "item": {
                "type": "command_execution",
                "status": "in_progress",
                "command": "python transcribe_audio.py clip.wav --model large-v3-4bit",
            },
        }
    )
    assert prepare == {
        "stage": "prepare_audio",
        "state": "active",
        "message": "آماده‌سازی، اندازه‌گیری و تقویت صدا",
    }
    assert whisper and whisper["stage"] == "whisper_review"
    assert "secret.mp3" not in prepare["message"]


def test_reads_only_new_jsonl_progress_lines(tmp_path):
    log = tmp_path / "run.jsonl"
    records = [
        {"type": "thread.started", "thread_id": "abc"},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "status": "completed",
                "exit_code": 0,
                "command": "ffprobe call.mp3",
            },
        },
    ]
    log.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    offset, events = read_progress_events(log)
    assert offset == 3
    assert len(events) == 3
    assert events[-1]["stage"] == "inspect_audio"
    assert read_progress_events(log, after_line=offset) == (3, [])
