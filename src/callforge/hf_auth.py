"""Interactive-only Hugging Face login; credentials never enter CLI arguments."""
import getpass
import os
import sys
from pathlib import Path


def main() -> int:
    if not sys.stdin.isatty():
        print("Run callforge hf-login in an interactive terminal.", file=sys.stderr)
        return 1
    from huggingface_hub import constants, login
    token = getpass.getpass("Hugging Face read token (hidden): ").strip()
    if not token:
        print("No token entered; nothing changed.")
        return 1
    try:
        login(token=token, add_to_git_credential=False)
        if os.name != "nt":
            for name in (constants.HF_TOKEN_PATH, constants.HF_STORED_TOKENS_PATH):
                path = Path(name)
                if path.is_file():
                    path.chmod(0o600)
    except Exception:
        print("Login failed. Check your token, read permissions and network connection.", file=sys.stderr)
        return 1
    finally:
        token = ""
    print("Hugging Face access saved in the CallForge model cache. No git credential was added.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
