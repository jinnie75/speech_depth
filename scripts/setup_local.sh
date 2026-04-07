#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
FRONTEND_DIR="$ROOT_DIR/frontend"
ENV_TEMPLATE="$ROOT_DIR/.env.local.example"
ENV_FILE="$ROOT_DIR/.env.local"
WORDNET_ZIP="$ROOT_DIR/.nltk_data/corpora/wordnet.zip"
SENTIWORDNET_ZIP="$ROOT_DIR/.nltk_data/corpora/sentiwordnet.zip"
NRC_EMOLEX="$ROOT_DIR/.lexicons/nrc_emolex/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

load_env_file() {
  local file="$1"
  if [ -f "$file" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$file"
    set +a
  fi
}

upsert_env_var() {
  local file="$1"
  local key="$2"
  local value="$3"
  local tmp

  tmp="$(mktemp)"
  if [ -f "$file" ]; then
    awk -v key="$key" -v value="$value" '
      BEGIN { updated = 0 }
      $0 ~ "^" key "=" {
        print key "=" value
        updated = 1
        next
      }
      { print }
      END {
        if (!updated) {
          print key "=" value
        }
      }
    ' "$file" > "$tmp"
  else
    printf '%s=%s\n' "$key" "$value" > "$tmp"
  fi
  mv "$tmp" "$file"
  chmod 600 "$file"
}

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found on PATH." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required but was not found on PATH." >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

if [ -f "$ENV_TEMPLATE" ] && [ ! -f "$ENV_FILE" ]; then
  echo "Creating .env.local from .env.local.example"
  cp "$ENV_TEMPLATE" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
fi

load_env_file "$ENV_FILE"

if [ -n "${HUGGINGFACE_TOKEN:-}" ]; then
  upsert_env_var "$ENV_FILE" "HUGGINGFACE_TOKEN" "$HUGGINGFACE_TOKEN"
fi

if [ -z "${HUGGINGFACE_TOKEN:-}" ] && [ -t 0 ]; then
  printf "Hugging Face token for diarization: "
  IFS= read -r -s HUGGINGFACE_TOKEN
  printf '\n'
  if [ -n "$HUGGINGFACE_TOKEN" ]; then
    upsert_env_var "$ENV_FILE" "HUGGINGFACE_TOKEN" "$HUGGINGFACE_TOKEN"
  fi
fi

echo "Installing backend dependencies"
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -e "$ROOT_DIR[transcription,diarization]"

if [ -f "$WORDNET_ZIP" ] && [ -f "$SENTIWORDNET_ZIP" ] && [ -f "$NRC_EMOLEX" ]; then
  echo "Verified bundled analysis lexicon resources"
else
  echo "Warning: analysis lexicon resources are missing; emotion-word matching may be incomplete." >&2
fi

if [ -f "$FRONTEND_DIR/.env.example" ] && [ ! -f "$FRONTEND_DIR/.env" ]; then
  echo "Creating frontend/.env from frontend/.env.example"
  cp "$FRONTEND_DIR/.env.example" "$FRONTEND_DIR/.env"
fi

echo "Installing frontend dependencies"
npm --prefix "$FRONTEND_DIR" install

cat <<EOF

Local setup is ready.

Next steps:
  ./scripts/run_api.sh
  ./scripts/run_worker.sh
EOF

if [ -z "${HUGGINGFACE_TOKEN:-}" ]; then
  echo
  echo "Add HUGGINGFACE_TOKEN to .env.local before running diarization."
fi
