from pathlib import Path

from callforge import cli
from callforge.registry import get_active_root, set_active_root, state_path


def test_registry_round_trip_uses_per_user_state(tmp_path: Path, monkeypatch):
    state_home = tmp_path / "state"
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    monkeypatch.setenv("CALLFORGE_HOME", str(state_home))

    assert set_active_root(audio_root) == audio_root.resolve()
    assert get_active_root() == audio_root.resolve()
    assert state_path() == state_home / "state.json"


def test_commands_use_active_workspace_without_current_directory(tmp_path: Path, monkeypatch, capsys):
    state_home = tmp_path / "state"
    audio_root = tmp_path / "audio"
    elsewhere = tmp_path / "elsewhere"
    audio_root.mkdir()
    elsewhere.mkdir()
    (audio_root / "external-201-123-20260419-193423-id.mp3").write_bytes(b"fake")
    monkeypatch.setenv("CALLFORGE_HOME", str(state_home))

    assert cli.main(["init", str(audio_root)]) == 0
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(cli, "scan", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected scan")))

    assert cli.main(["workspace"]) == 0
    assert cli.main(["status"]) == 0
    assert cli.main(["run", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert f"Audio directory: {audio_root.resolve()}" in output
    assert "audio_files: 1" in output
    assert "Dry run: would claim up to 1 of 1 pending jobs" in output


def test_scan_switches_the_active_workspace(tmp_path: Path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv("CALLFORGE_HOME", str(tmp_path / "state"))

    assert cli.main(["init", str(first)]) == 0
    assert get_active_root() == first.resolve()
    assert cli.main(["scan", str(second)]) == 0
    assert get_active_root() == second.resolve()


def test_workspace_command_explains_how_to_select_audio_directory(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("CALLFORGE_HOME", str(tmp_path / "empty-state"))
    assert cli.main(["workspace"]) == 1
    assert "callforge init /path/to/audio" in capsys.readouterr().err


def test_only_commands_that_need_a_scope_accept_a_directory_argument():
    parser = cli.build_parser()
    assert parser.parse_args(["init", "/audio"]).directory == "/audio"
    assert parser.parse_args(["scan", "/audio"]).directory == "/audio"
    assert parser.parse_args(["reset", "/audio/archive"]).directory == "/audio/archive"
    for command in ("run", "ui", "status", "retry", "transcripts", "workspace"):
        assert not hasattr(parser.parse_args([command]), "directory")


def test_reset_command_uses_active_workspace_and_requires_confirmation(tmp_path: Path, monkeypatch, capsys):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    audio = audio_root / "external-201-123-20260419-193423-id.mp3"
    audio.write_bytes(b"audio")
    monkeypatch.setenv("CALLFORGE_HOME", str(tmp_path / "state"))
    assert cli.main(["init", str(audio_root)]) == 0

    monkeypatch.setattr("builtins.input", lambda _: "cancel")
    assert cli.main(["reset"]) == 1
    _, database = cli.workspace()
    assert database.reset_count() == 1

    assert cli.main(["reset", "--yes"]) == 0
    assert database.reset_count() == 0
    assert audio.is_file()
    assert "Source files were not changed" in capsys.readouterr().out
