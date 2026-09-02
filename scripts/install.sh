#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
ENV_DIR=${CALLFORGE_ENV_DIR:-"$PROJECT_DIR/.venv"}

PYTHON_BIN=${PYTHON_BIN:-python3}
"$PYTHON_BIN" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install "$PROJECT_DIR"
"$ENV_DIR/bin/callforge" setup --yes

printf '%s\n' "CallForge installed at $ENV_DIR/bin/callforge"

