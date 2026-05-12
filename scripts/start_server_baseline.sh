#!/usr/bin/env bash
# =============================================================================
# start_server_baseline.sh
#
# Launches llama-server for baseline (standard autoregressive) decoding.
#
# Port: 8000
# Backend: llama.cpp (TurboQuant fork, built with sm_70 for V100)
#
# Prereqs:
#   1. bash scripts/build_llamacpp.sh         (build once)
#   2. export PATH="$HOME/llama-cpp-turboquant/build/bin:$PATH"
#   3. bash scripts/download_models.sh        (download once)
#   4. .env with TARGET_MODEL_GGUF and MODEL_ALIAS set
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

# Load .env
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

TARGET_FILE="${TARGET_MODEL_GGUF:-$REPO_ROOT/models/gemma-4-E4B-it-Q8_0.gguf}"
MODEL_ALIAS="${MODEL_ALIAS:-gemma-4-e4b-it}"
CONTEXT_LEN="${CONTEXT_LEN:-32768}"
# -1 = no limit on generated tokens (honour max_tokens from the API request)
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
PORT=8000
LOG_FILE="$REPO_ROOT/experiments/logs/baseline_server.log"

mkdir -p "$REPO_ROOT/experiments/logs"

# -----------------------------------------------------------------------
# Prereq checks
# -----------------------------------------------------------------------
if [ ! -f "$TARGET_FILE" ]; then
    echo "[ERROR] GGUF file not found: $TARGET_FILE"
    echo "        Run: bash scripts/download_models.sh"
    exit 1
fi

if ! command -v llama-server &> /dev/null; then
    echo "[ERROR] llama-server not in PATH."
    echo "        Run: bash scripts/build_llamacpp.sh"
    echo "        Then: export PATH=\"\$HOME/llama-cpp-turboquant/build/bin:\$PATH\""
    exit 1
fi

if lsof -Pi :"$PORT" -sTCP:LISTEN -t > /dev/null 2>&1; then
    echo "[ERROR] Port $PORT is already in use."
    exit 1
fi

echo "========================================================"
echo " Starting BASELINE llama-server"
echo " Model:   $(basename "$TARGET_FILE")"
echo " Alias:   $MODEL_ALIAS"
echo " Context: $CONTEXT_LEN tokens"
echo " Port:    $PORT"
echo " Log:     $LOG_FILE"
echo "========================================================"

llama-server \
    --model         "$TARGET_FILE" \
    --alias         "$MODEL_ALIAS" \
    --ctx-size      "$CONTEXT_LEN" \
    --n-gpu-layers  "$N_GPU_LAYERS" \
    --reasoning-budget 0 \
    --flash-attn on \
    --cache-type-k  q8_0 \
    --cache-type-v  q8_0 \
    --cont-batching \
    --chat-template-kwargs '{"enable_thinking":false}' # for Gemma-4-E4B
    --port          "$PORT" \
    --host          127.0.0.1 \
    2>&1 | grep -E "(llama_|Server|HTTP|POST|ready|health|Error|error|done request|print_timing|release|slots are idle|Chat format|FAIL|PASS|Starting|Model|Port)" | tee "$LOG_FILE" &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo "$SERVER_PID" > "$REPO_ROOT/experiments/logs/baseline_server.pid"

echo ""
echo "Waiting for server to become ready …"
for i in $(seq 1 60); do
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "[OK] Baseline server is live at http://localhost:$PORT/v1"
        echo ""
        curl -s "http://localhost:$PORT/v1/models" | python3 -m json.tool 2>/dev/null || true
        exit 0
    fi
    echo "  ($i/60) waiting …"
    sleep 5
done

echo "[TIMEOUT] Server did not respond after 5 minutes."
echo "Check $LOG_FILE for errors."
exit 1
