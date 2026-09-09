import json
import sys
from types import SimpleNamespace

import pytest

from callforge.asr_benchmark import run_comparison
from callforge.config import AppConfig
from callforge.quality import file_hash


def test_asr_experiment_keeps_reference_out_of_decoder_and_preserves_files(tmp_path, monkeypatch):
    source = tmp_path / "call.mp3"
    source.write_bytes(b"audio")
    markdown = source.with_suffix(".md")
    markdown.write_text("approved")
    snapshot = tmp_path / "reference.json"
    snapshot.write_text(json.dumps({"schema_version": 1, "split": "development", "audio_id": 1,
        "audio_path": str(source), "audio_sha256": file_hash(source),
        "reference": {"content": "", "segments": [{"text": "SECRET REFERENCE"}]}}))
    before = [file_hash(path) for path in (source, markdown, snapshot)]
    downloads, commands = [], []
    def download(**kwargs):
        downloads.append(kwargs)
        return "/cached/fixed-revision"
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    def run(self, command, destination, *args, **kwargs):
        commands.append(command)
        if command[1].endswith("prepare_audio.py"):
            return {"raw_wav": "raw.wav", "agc_wav": "agc.wav", "duration_seconds": 2}
        return {"segments": [{"start": 0, "end": 1, "text": "سلام"}], "elapsed_seconds": 3}
    monkeypatch.setattr("callforge.asr_benchmark.LocalWhisperPipeline._run_json", run)
    config = AppConfig.for_root(tmp_path)
    result = run_comparison(config, [snapshot], tmp_path / "experiment", ["model-a", "model-b"],
                            temperatures=[0, .2, .4], seed=0, context_prompt=False)
    assert result["status"] == "completed"
    assert len(result["results"]) == 4
    assert len(commands) == 5
    assert commands[1][commands[1].index("--seed") + 1] == "0"
    assert commands[1][commands[1].index("--prompt") + 1] == ""
    assert commands[1].count("--temperature") == 3
    assert all("SECRET" not in " ".join(command) for command in commands)
    assert downloads[0]["cache_dir"] == str(config.models / "hub")
    assert before == [file_hash(path) for path in (source, markdown, snapshot)]
    assert not config.database.exists()
    with pytest.raises(FileExistsError):
        run_comparison(config, [snapshot], tmp_path / "experiment", ["model-a"])
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="no longer matches"):
        run_comparison(config, [snapshot], tmp_path / "another", ["model-a"])
    assert not (tmp_path / "another").exists()
