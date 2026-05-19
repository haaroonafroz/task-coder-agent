#!/usr/bin/env bash
# =============================================================================
# start_server_baseline.sh
#
# Launches llama-server for baseline (standard autoregressive) decoding.
# No speculative decoding — compare against start_server_speculative.sh.
#
# Port: 8000
# Model: from .env (TARGET_MODEL_GGUF, MODEL_ALIAS) — default Qwen 3.6
#
# Prereqs:
#   1. Build llama.cpp with CUDA (see docs/build.md)
#   2. export PATH="$REPO_ROOT/llama.cpp/build/bin:$PATH"
#   3. bash scripts/download_models.sh --qwen36
#   4. .env with TARGET_MODEL_GGUF and MODEL_ALIAS
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

TARGET_FILE="${TARGET_MODEL_GGUF:-$REPO_ROOT/models/Qwen3.6-27B-UD-Q4_K_XL.gguf}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen36-27b-mtp}"
CONTEXT_LEN="${CONTEXT_LEN:-8192}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
PORT=8000
LOG_FILE="$REPO_ROOT/experiments/logs/baseline_server.log"

mkdir -p "$REPO_ROOT/experiments/logs"

if [ ! -f "$TARGET_FILE" ]; then
    echo "[ERROR] GGUF file not found: $TARGET_FILE"
    echo "        Run: bash scripts/download_models.sh --qwen36"
    exit 1
fi

if ! command -v llama-server &> /dev/null; then
    echo "[ERROR] llama-server not in PATH."
    echo "        export PATH=\"$REPO_ROOT/llama.cpp/build/bin:\$PATH\""
    exit 1
fi

if lsof -Pi :8001 -sTCP:LISTEN -t > /dev/null 2>&1; then
    echo "[WARNING] Speculative server is running on port 8001."
    echo "          Running both may exceed 32GB VRAM on 2x V100."
    read -r -p "Continue anyway? [y/N] " CONT
    if [[ "${CONT,,}" != "y" ]]; then
        echo "Aborting. Stop speculative first: bash scripts/stop_server.sh speculative"
        exit 1
    fi
fi

if lsof -Pi :"$PORT" -sTCP:LISTEN -t > /dev/null 2>&1; then
    echo "[ERROR] Port $PORT is already in use."
    exit 1
fi

echo "========================================================"
echo " Starting BASELINE llama-server (no speculative decoding)"
echo " Model:   $(basename "$TARGET_FILE")"
echo " Alias:   $MODEL_ALIAS"
echo " Context: $CONTEXT_LEN tokens"
echo " GPUs:    CUDA0,CUDA1 (layer split)"
echo " Port:    $PORT"
echo " Log:     $LOG_FILE"
echo "========================================================"

llama-server \
    --model "$TARGET_FILE" \
    --alias "$MODEL_ALIAS" \
    --ctx-size "$CONTEXT_LEN" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --device CUDA0,CUDA1 \
    --split-mode layer \
    --tensor-split 1,1 \
    --flash-attn on \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --cont-batching \
    --port "$PORT" \
    --host 127.0.0.1 \
    --log-verbosity 2 \
    2>&1 | tee "$LOG_FILE" &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo "$SERVER_PID" > "$REPO_ROOT/experiments/logs/baseline_server.pid"

echo ""
echo "Waiting for server to become ready …"
for i in $(seq 1 90); do
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "[OK] Baseline server is live at http://localhost:$PORT/v1"
        echo ""
        curl -s "http://localhost:$PORT/v1/models" | python3 -m json.tool 2>/dev/null || true
        exit 0
    fi
    echo "  ($i/90) waiting …"
    sleep 5
done

echo "[TIMEOUT] Server did not respond after 7.5 minutes."
echo "Check $LOG_FILE for errors."
exit 1