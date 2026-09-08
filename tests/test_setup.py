import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from callforge.config import AppConfig
from callforge.registry import callforge_home, set_active_root
from callforge import speaker_runtime, setup_tools


def test_global_setup_paths_work_before_init_and_preserve_legacy_cache(tmp_path):
    runtime, models = speaker_runtime.paths()
    assert runtime.parent == callforge_home()
    speaker_runtime.register()
    config = AppConfig.for_root(tmp_path)
    config.ensure()
    assert config.diarization is True
    assert config.runtime_environment()["HF_HOME"] == str(models)
    (config.models / "hub").mkdir()
    assert config.runtime_environment()["HF_HOME"] == str(config.models)
    assert speaker_runtime.environment(offline=True)["HF_HUB_OFFLINE"] == "1"
    assert speaker_runtime.environment()["DISABLE_SAFETENSORS_CONVERSION"] == "1"


def test_setup_reuses_existing_runtime_and_enables_only_target_setting(tmp_path):
    config = AppConfig.for_root(tmp_path)
    config.ensure()
    (config.workspace / "diarization-runtime").mkdir()
    set_active_root(tmp_path)
    assert speaker_runtime.paths() == (config.workspace / "diarization-runtime", config.models)
    path = config.workspace / "config.toml"
    path.write_text('[callforge]\nworkers = 7\ndiarization = false # keep comment\n[unrelated]\ndiarization = false\n')
    setup_tools.enable_speakers(config)
    assert path.read_text() == '[callforge]\nworkers = 7\ndiarization = true # keep comment\n[unrelated]\ndiarization = false\n'


@pytest.mark.parametrize("access,interactive,expected_login,success", [
    (0, False, False, True), (2, False, False, False), (2, True, True, True),
    (3, True, True, True), (4, True, False, True), (5, True, False, False)])
def test_setup_access_flow_no_token_arguments_no_false_success(tmp_path, monkeypatch, capsys, access, interactive, expected_login, success):
    def install():
        speaker_runtime.python().parent.mkdir(parents=True)
        speaker_runtime.python().touch()
        speaker_runtime.register()
    monkeypatch.setattr(speaker_runtime, "install", install)
    monkeypatch.setattr("sys.stdin.isatty", lambda: interactive)
    monkeypatch.setattr("builtins.input", lambda _: "")
    calls, accesses = [], []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[-1] == "access":
            accesses.append(True)
            return SimpleNamespace(returncode=access if len(accesses) == 1 else 0)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(setup_tools.subprocess, "run", run)
    if success:
        setup_tools.setup_models()
        assert setup_tools.speaker_check().ok
        assert any(c[-1] == "models" and kw["env"].get("HF_HUB_OFFLINE") == "1" for c, kw in calls)
        assert any(c[-1] == "whisper" for c, _ in calls)
    else:
        with pytest.raises(RuntimeError):
            setup_tools.setup_models()
        assert not setup_tools.speaker_check().ok
        assert not any(c[-1] == "models" for c, _ in calls)
    assert any(c[-1] == "callforge.hf_auth" for c, _ in calls) == expected_login
    assert all("--token" not in c for c, _ in calls)
    if access:
        out = capsys.readouterr().out
        assert "https://huggingface.co/settings/tokens" in out
        assert "https://huggingface.co/pyannote/speaker-diarization-community-1" in out


def test_model_failure_never_marks_setup_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(speaker_runtime, "install", lambda: None)
    def run(command, **kwargs):
        if command[-1] == "models":
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(setup_tools.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        setup_tools.setup_models()
    assert not setup_tools.speaker_check().ok
