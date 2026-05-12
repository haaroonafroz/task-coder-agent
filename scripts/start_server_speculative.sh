#!/usr/bin/env bash
# =============================================================================
# start_server_speculative.sh - Qwen 3.6 MTP Configuration
#
# Qwen 3.6 has NATIVE MTP - no separate draft model needed!
# The model itself generates speculative tokens internally.
# =============================================================================

set -euo pipefail

MODEL="Qwen/Qwen3.6-35B-A3B"  # 35B total, 3B active params (MoE)
NUM_SPEC_TOKENS=3              # Start with 2, can increase to 3-5
PORT=8001
GPU_MEM_UTIL=0.95               # Slightly higher for MoE efficiency
MAX_MODEL_LEN=32768              # Qwen supports longer contexts
LOG_FILE="experiments/logs/speculative_server.log"

mkdir -p experiments/logs

echo "========================================================"
echo " Starting Qwen 3.6 SPECULATIVE (Native MTP) vLLM server"
echo " Model:  $MODEL"
echo " Port:   $PORT"
echo " Spec tokens: $NUM_SPEC_TOKENS"
echo " GPU util: $GPU_MEM_UTIL"
echo " Log:    $LOG_FILE"
echo "========================================================"

# Safety check
if lsof -Pi :"$PORT" -sTCP:LISTEN -t > /dev/null 2>&1; then
    echo "[ERROR] Port $PORT is already in use."
    exit 1
fi

# Qwen 3.6 native MTP config - NOTICE: no "model" field needed!
SPEC_CONFIG="{\"method\":\"mtp\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS}"
BNB_CONFIG='{"load_in_4bit":true,"bnb_4bit_compute_dtype":"float16","bnb_4bit_quant_type":"nf4","bnb_4bit_use_double_quant":true}'

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --speculative-config "$SPEC_CONFIG" \
    --dtype float16 \
    --quantization bitsandbytes  \
    --model-loader-extra-config '{"load_in_4bit":true,"bnb_4bit_compute_dtype":"float16","bnb_4bit_quant_type":"nf4"}' \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --port "$PORT" \
    2>&1 | tee "$LOG_FILE" &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo "$SERVER_PID" > experiments/logs/speculative_server.pid

echo ""
echo "Waiting for server to become ready ..."
for i in $(seq 1 90); do
    if curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
        echo "[OK] Speculative server is live at http://localhost:$PORT/v1"
        echo ""
        echo "Verifying MTP is active in logs ..."
        if grep -q "MTP\|speculative" "$LOG_FILE" 2>/dev/null; then
            echo "[OK] MTP confirmed active in logs."
        else
            echo "[INFO] Check $LOG_FILE for speculative decoding activity."
        fi
        exit 0
    fi
    echo "  ($i/90) waiting ..."
    sleep 5
done

echo "[TIMEOUT] Server did not respond after 7.5 minutes."
exit 1