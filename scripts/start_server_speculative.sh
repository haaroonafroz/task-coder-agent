#!/usr/bin/env bash
# =============================================================================
# start_server_speculative.sh
#
# Launches llama-server with MTP speculative decoding.
# The draft model (-md) is the purpose-built assistant for the target.
#
# Port: 8001
# Backend: llama.cpp (TurboQuant fork, built with sm_70 for V100)
#
# IMPORTANT: Stop the baseline server before starting this one.
#            Running both simultaneously exceeds 2x16GB VRAM on V100s.
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
# DRAFT_FILE="${DRAFT_MODEL_GGUF:-$REPO_ROOT/models/gemma-4-E4B-it-assistant-F16.gguf}"
MODEL_ALIAS="${MODEL_ALIAS:-gemma-4-e4b-it}"
CONTEXT_LEN="${CONTEXT_LEN:-8192}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
NUM_DRAFT_TOKENS="${NUM_DRAFT_TOKENS:-4}"
PORT=8001
LOG_FILE="$REPO_ROOT/experiments/logs/speculative_server.log"

mkdir -p "$REPO_ROOT/experiments/logs"

# -----------------------------------------------------------------------
# Prereq checks
# -----------------------------------------------------------------------
if [ ! -f "$TARGET_FILE" ]; then
    echo "[ERROR] Target GGUF not found: $TARGET_FILE"
    echo "        Run: bash scripts/download_models.sh"
    exit 1
fi

# if [ ! -f "$DRAFT_FILE" ]; then
#     echo "[ERROR] Draft GGUF not found: $DRAFT_FILE"
#     echo "        Run: bash scripts/download_models.sh"
#     exit 1
# fi

if ! command -v llama-server &> /dev/null; then
    echo "[ERROR] llama-server not in PATH."
    echo "        Run: bash scripts/build_llamacpp.sh"
    echo "        Then: export PATH=\"\$HOME/llama-cpp-turboquant/build/bin:\$PATH\""
    exit 1
fi

# Warn if baseline server is still running
if lsof -Pi :8000 -sTCP:LISTEN -t > /dev/null 2>&1; then
    echo "[WARNING] Baseline server is still running on port 8000."
    echo "          Running both simultaneously may exceed 32GB VRAM on 2x V100."
    read -r -p "Continue anyway? [y/N] " CONT
    if [[ "${CONT,,}" != "y" ]]; then
        echo "Aborting. Stop baseline server first: bash scripts/stop_server.sh baseline"
        exit 1
    fi
fi

if lsof -Pi :"$PORT" -sTCP:LISTEN -t > /dev/null 2>&1; then
    echo "[ERROR] Port $PORT is already in use."
    exit 1
fi

echo "========================================================"
echo " Starting SPECULATIVE llama-server"
echo " Target:  $(basename "$TARGET_FILE")"
# echo " Draft:   $(basename "$DRAFT_FILE")"
echo " Alias:   $MODEL_ALIAS"
echo " Context: $CONTEXT_LEN tokens"
echo " Draft tokens: $NUM_DRAFT_TOKENS"
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
    --spec-type draft-mtp \
    --spec-draft-n-max "$NUM_DRAFT_TOKENS" \
    --port "$PORT" \
    --host 127.0.0.1 \
    --log-file "$LOG_FILE" \
    --log-verbosity 3 &

SERVER_PID=$!
echo "$SERVER_PID" > "$REPO_ROOT/experiments/logs/speculative_server.pid"

echo ""
echo "Waiting for server to become ready …"
for i in $(seq 1 90); do
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "[OK] Speculative server is live at http://localhost:$PORT/v1"
        echo ""
        echo "Verifying draft model is loaded …"
        if grep -qiE "draft model|model-draft|speculative" "$LOG_FILE" 2>/dev/null; then
            echo "[OK] Draft model confirmed in logs."
        else
            echo "[WARNING] Draft model not confirmed in logs yet."
            echo "          Check $LOG_FILE manually."
        fi
        curl -s "http://localhost:$PORT/v1/models" | python3 -m json.tool 2>/dev/null || true
        exit 0
    fi
    echo "  ($i/90) waiting …"
    sleep 5
done

echo "[TIMEOUT] Server did not respond after 7.5 minutes."
echo "Check $LOG_FILE for errors."
exit 1
