#!/usr/bin/env sh
set -eu

PYTHON_BIN=${PYTHON_BIN:-python3}
CALLFORGE_SOURCE=${CALLFORGE_SOURCE:-git+https://github.com/oveysrostami/callforge.git}
INSTALL_ROOT=${CALLFORGE_INSTALL_DIR:-"${XDG_DATA_HOME:-$HOME/.local/share}/callforge"}
ENV_DIR="$INSTALL_ROOT/venv"
BIN_DIR=${CALLFORGE_BIN_DIR:-"$HOME/.local/bin"}

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
    printf '%s\n' "CallForge requires Python 3.11 or newer. Set PYTHON_BIN to a compatible interpreter." >&2
    exit 1
fi

mkdir -p "$INSTALL_ROOT" "$BIN_DIR"
if [ ! -x "$ENV_DIR/bin/python" ]; then
    "$PYTHON_BIN" -m venv "$ENV_DIR"
fi
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install --upgrade --force-reinstall "$CALLFORGE_SOURCE"
"$ENV_DIR/bin/python" -m callforge setup --yes --force-skill
ln -sfn "$ENV_DIR/bin/callforge" "$BIN_DIR/callforge"

PATH_LINE="export PATH=\"$BIN_DIR:\$PATH\""
case "${SHELL:-}" in
    */zsh) PROFILE_FILE="$HOME/.zprofile" ;;
    */bash) PROFILE_FILE="$HOME/.bash_profile" ;;
    *) PROFILE_FILE="$HOME/.profile" ;;
esac
if [ ! -f "$PROFILE_FILE" ] || ! grep -F "$PATH_LINE" "$PROFILE_FILE" >/dev/null 2>&1; then
    printf '\n%s\n' "$PATH_LINE" >> "$PROFILE_FILE"
fi

printf '%s\n' "CallForge installed: $BIN_DIR/callforge"
printf '%s\n' "Open a new terminal once if 'callforge' is not yet on PATH."
printf '%s\n' "Next: callforge init /path/to/audio"
