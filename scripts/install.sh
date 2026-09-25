#!/usr/bin/env bash
# Native install for the Missions harness (Linux). Inference is NOT installed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
TASK_CODER_HOME="${TASK_CODER_HOME:-$HOME/.missions}"

echo "==> Missions native install"
echo "    repo: $REPO_ROOT"
echo "    home: $TASK_CODER_HOME"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "[ERROR] python3 not found"
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    echo "[ERROR] git not found"
    exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
    echo "[WARN] npm not found — UI will not be built. Install Node 18+ and re-run."
fi

if ! command -v bwrap >/dev/null 2>&1; then
    echo "[WARN] bubblewrap (bwrap) not found."
    echo "       External workspaces will run with policy checks only."
    echo "       Debian/Ubuntu: sudo apt install bubblewrap"
fi

if [ ! -d "$REPO_ROOT/.venv" ]; then
    "$PYTHON" -m venv "$REPO_ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"
pip install -U pip
pip install -e "$REPO_ROOT"

if [ "${SKIP_PLAYWRIGHT:-0}" != "1" ]; then
    python -m playwright install chromium || echo "[WARN] playwright chromium install failed"
fi

if command -v npm >/dev/null 2>&1; then
    npm --prefix "$REPO_ROOT/frontend" ci
    npm --prefix "$REPO_ROOT/frontend" run build
fi

export TASK_CODER_HOME
missions init
missions doctor

echo
echo "Start the harness:"
echo "  source $REPO_ROOT/.venv/bin/activate"
echo "  export TASK_CODER_HOME=\"$TASK_CODER_HOME\""
echo "  missions serve"
echo
echo "Then open http://127.0.0.1:8088"
echo "Point Settings at your local /v1 server or a cloud API key."
echo "llama.cpp / Ollama / LM Studio are started separately."
