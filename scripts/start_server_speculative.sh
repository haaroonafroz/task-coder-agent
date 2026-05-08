#!/usr/bin/env bash
# =============================================================================
# start_server_speculative.sh
#
# Launches the speculative (MTP) vLLM server.
# Section 6.3 of the implementation plan.
#
# Port: 8001
# Target: google/gemma-4-E4B-it
# Draft:  google/gemma-4-E4B-it-assistant   (the ONLY correct choice)
# Speculative tokens: 4
#
# IMPORTANT: Run ONLY after the baseline server is fully stopped.
# Running both simultaneously will exceed 2x16GB VRAM budget.
# =============================================================================

set -euo pipefail

MODEL="google/gemma-4-E4B-it"
DRAFT_MODEL="google/gemma-4-E4B-it-assistant"
NUM_SPEC_TOKENS=4
PORT=8001
GPU_MEM_UTIL=0.85
MAX_MODEL_LEN=4096
LOG_FILE="experiments/logs/speculative_server.log"

mkdir -p experiments/logs

echo "========================================================"
echo " Starting SPECULATIVE (MTP) vLLM server"
echo " Target: $MODEL"
echo " Draft:  $DRAFT_MODEL"
echo " Spec tokens: $NUM_SPEC_TOKENS"
echo " Port:   $PORT"
echo " Quant:  bitsandbytes 8-bit"
echo " Log:    $LOG_FILE"
echo "========================================================"

# Safety check: ensure baseline server is not running
if lsof -Pi :8000 -sTCP:LISTEN -t > /dev/null 2>&1; then
    echo "[WARNING] Baseline server is still running on port 8000."
    echo "          Running both simultaneously may exceed 32GB VRAM."
    read -r -p "Continue anyway? [y/N] " CONT
    if [[ "${CONT,,}" != "y" ]]; then
        echo "Aborting. Stop baseline server first."
        exit 1
    fi
fi

# Verify port is free
if lsof -Pi :"$PORT" -sTCP:LISTEN -t > /dev/null 2>&1; then
    echo "[ERROR] Port $PORT is already in use. Kill existing process first."
    exit 1
fi

# Build speculative-config JSON
SPEC_CONFIG="{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS}"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --speculative-config "$SPEC_CONFIG" \
    --dtype bfloat16 \
    --quantization bitsandbytes \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-chunked-prefill \
    --port "$PORT" \
    2>&1 | tee "$LOG_FILE" &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo "$SERVER_PID" > experiments/logs/speculative_server.pid

echo ""
echo "Waiting for server to become ready …"
for i in $(seq 1 90); do
    if curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
        echo "[OK] Speculative server is live at http://localhost:$PORT/v1"
        echo ""
        echo "Verifying MTP is active in logs …"
        if grep -q "Gemma4 MTP" "$LOG_FILE" 2>/dev/null; then
            echo "[OK] MTP confirmed active (found 'Gemma4 MTP' in logs)."
        else
            echo "[WARNING] 'Gemma4 MTP' not found in logs yet."
            echo "          Check $LOG_FILE for:"
            echo "            'Gemma4 MTP: centroids masking enabled'"
            echo "            'Gemma4 MTP: draft layer N -> target_layer...'"
            echo "          If absent, speculative decoding is NOT active."
        fi
        exit 0
    fi
    echo "  ($i/90) waiting …"
    sleep 5
done

echo "[TIMEOUT] Server did not respond after 7.5 minutes."
echo "Check $LOG_FILE for errors."
exit 1
