#!/usr/bin/env bash
# =============================================================================
# start_server_baseline.sh
#
# Launches the baseline vLLM server (standard autoregressive decoding).
# Section 6.2 of the implementation plan.
#
# Port: 8000
# Model: google/gemma-4-E4B-it
# Quantization: bitsandbytes 8-bit
# =============================================================================

set -euo pipefail

MODEL="google/gemma-4-E4B-it"
PORT=8000
GPU_MEM_UTIL=0.85
MAX_MODEL_LEN=4096
LOG_FILE="experiments/logs/baseline_server.log"

mkdir -p experiments/logs

echo "========================================================"
echo " Starting BASELINE vLLM server"
echo " Model:  $MODEL"
echo " Port:   $PORT"
echo " Quant:  bitsandbytes 8-bit"
echo " Log:    $LOG_FILE"
echo "========================================================"

# Verify no other vLLM process is already running on this port
if lsof -Pi :"$PORT" -sTCP:LISTEN -t > /dev/null 2>&1; then
    echo "[ERROR] Port $PORT is already in use. Kill existing process first."
    exit 1
fi

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype bfloat16 \
    --quantization bitsandbytes \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-chunked-prefill \
    --port "$PORT" \
    2>&1 | tee "$LOG_FILE" &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo "$SERVER_PID" > experiments/logs/baseline_server.pid

echo ""
echo "Waiting for server to become ready …"
for i in $(seq 1 60); do
    if curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
        echo "[OK] Baseline server is live at http://localhost:$PORT/v1"
        echo "     curl http://localhost:$PORT/v1/models"
        curl -s "http://localhost:$PORT/v1/models" | python -m json.tool 2>/dev/null || true
        exit 0
    fi
    echo "  ($i/60) waiting …"
    sleep 5
done

echo "[TIMEOUT] Server did not respond after 5 minutes."
echo "Check $LOG_FILE for errors."
exit 1
