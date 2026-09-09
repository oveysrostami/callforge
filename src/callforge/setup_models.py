"""Fresh-process setup probes: HF_HOME must be set before importing model libraries."""
from __future__ import annotations

import argparse
import os
import sys

COMMUNITY = "pyannote/speaker-diarization-community-1"


def access() -> int:
    from huggingface_hub import HfApi, get_token, hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    token = get_token()
    if not token:
        return 2
    try:
        HfApi().whoami(token=token)
        # Cached config alone cannot prove that this account accepted the gate.
        HfApi().auth_check(repo_id=COMMUNITY, token=token)
        hf_hub_download(COMMUNITY, "config.yaml", token=token)
    except GatedRepoError:
        return 4
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        return 3 if status == 401 else (4 if status == 403 else 5)
    except Exception:
        return 5
    return 0


def models() -> None:
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    from pyannote.audio import Pipeline
    print("Downloading/loading community-1 locally...", flush=True)
    pipeline = Pipeline.from_pretrained(COMMUNITY, token=os.environ.get("HF_TOKEN"))
    if pipeline is None:
        raise RuntimeError("Model access unavailable")
    del pipeline
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from callforge.alignment import MODEL, REVISION
    print("Downloading/loading Persian word-alignment model (~1.26 GB on first setup)...", flush=True)
    Wav2Vec2Processor.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    Wav2Vec2ForCTC.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False,
                                  weights_only=True, use_safetensors=False).eval()
    print("Speaker and alignment models loaded successfully.", flush=True)


def whisper() -> None:
    import importlib.util
    from contextlib import ExitStack
    from pathlib import Path
    helper = Path(__file__).parent / "resources/pbx-call-transcriber/scripts/transcribe_audio.py"
    spec = importlib.util.spec_from_file_location("callforge_setup_asr", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("Downloading/loading Whisper turbo (existing cache is reused)...", flush=True)
    if module.choose_backend("auto") == "mlx":
        from mlx_whisper.load_models import load_model
        with ExitStack() as stack:
            model, _ = module.compatible_mlx_model(module.MLX_MODELS["turbo"], stack)
            load_model(model)
    else:
        from faster_whisper import WhisperModel
        WhisperModel("large-v3-turbo", device="auto", compute_type="default")
    from faster_whisper.vad import get_vad_model
    get_vad_model()
    print("Whisper and speech detector loaded successfully.", flush=True)


def quality_models() -> None:
    """Download every optional quality candidate; never called at runtime."""
    import importlib.util
    from contextlib import ExitStack
    from pathlib import Path
    helper = Path(__file__).parent / "resources/pbx-call-transcriber/scripts/transcribe_audio.py"
    spec = importlib.util.spec_from_file_location("callforge_setup_quality_asr", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("Downloading/loading non-Q4 Turbo and full large-v3...", flush=True)
    if module.choose_backend("auto") == "mlx":
        from mlx_whisper.load_models import load_model
        for name in ("turbo", "full"):
            with ExitStack() as stack:
                model, _ = module.compatible_mlx_model(module.MLX_MODELS[name], stack)
                load_model(model)
    else:
        from faster_whisper import WhisperModel
        for name in ("large-v3-turbo", "large-v3"):
            WhisperModel(name, device="auto", compute_type="default")
    from huggingface_hub import snapshot_download
    for model in ("Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B"):
        print(f"Downloading optional benchmark candidate {model}...", flush=True)
        snapshot_download(repo_id=model)
    print("Quality model cache is ready.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("access", "models", "whisper", "quality-models"))
    args = parser.parse_args()
    try:
        if args.mode == "access":
            return access()
        {"models": models, "whisper": whisper, "quality-models": quality_models}[args.mode]()
        return 0
    except Exception:
        # Never print an exception that could contain credentials or signed URLs.
        print("Model setup failed. Check model access, network, free disk space and runtime compatibility; rerun callforge setup.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
