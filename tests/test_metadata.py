from pathlib import Path

from callforge.metadata import extract_audio_metadata, parse_call_filename


def test_external_filename_metadata():
    parsed = parse_call_filename(
        "external-201-09013663769-20260419-193423-1776614663.1203835.mp3"
    )
    assert parsed == {
        "direction": "inbound",
        "agent_extension": "201",
        "remote_number": "09013663769",
        "call_id": "1776614663.1203835",
        "recorded_at": "2026-04-19T19:34:23",
    }


def test_internal_filename_metadata():
    parsed = parse_call_filename("internal-208-12345-20260408-143257-id.mp3")
    assert parsed == {
        "direction": "internal",
        "agent_extension": "208",
        "remote_number": "12345",
        "call_id": "id",
        "recorded_at": "2026-04-08T14:32:57",
    }


def test_out_filename_metadata():
    parsed = parse_call_filename(
        "out-09124975161-202-20260420-131200-1776678120.1233910.mp3"
    )
    assert parsed == {
        "direction": "outbound",
        "agent_extension": "202",
        "remote_number": "09124975161",
        "call_id": "1776678120.1233910",
        "recorded_at": "2026-04-20T13:12:00",
    }


def test_broken_mp3_is_still_indexable(tmp_path: Path):
    audio = tmp_path / "external-208-123-20260408-143257-id.mp3"
    audio.write_bytes(b"not an mp3")
    metadata = extract_audio_metadata(audio, tmp_path)
    assert metadata.size_bytes == 10
    assert metadata.metadata_error
    assert len(metadata.content_sha256) == 64
