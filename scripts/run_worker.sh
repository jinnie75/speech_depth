#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
ENV_FILE="$ROOT_DIR/.env.local"

if [ ! -x "$VENV_PYTHON" ]; then
  echo "Missing $VENV_PYTHON. Run ./scripts/setup_local.sh first." >&2
  exit 1
fi

cd "$ROOT_DIR"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$VENV_PYTHON" -m asr_viz.worker "$@"
