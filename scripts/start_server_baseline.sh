#!/usr/bin/env bash
set -euo pipefail

MODEL="QuantTrio/Qwen3.6-35B-A3B-AWQ"  # or "Qwen/Qwen3.6-35B-A3B"
PORT=8000
GPU_MEM_UTIL=0.90
MAX_MODEL_LEN=32768
LOG_FILE="experiments/logs/baseline_server.log"

mkdir -p experiments/logs

echo "========================================================"
echo " Starting Qwen 3.6 BASELINE (RTX 4090)"
echo " Model:  $MODEL"
echo " Port:   $PORT"
echo "========================================================"

if lsof -Pi :"$PORT" -sTCP:LISTEN -t > /dev/null 2>&1; then
    echo "[ERROR] Port $PORT is already in use."
    exit 1
fi

BNB_CONFIG='{"load_in_4bit":true,"bnb_4bit_compute_dtype":"bfloat16","bnb_4bit_quant_type":"nf4","bnb_4bit_use_double_quant":true}'

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype bfloat16 \
    --quantization bitsandbytes \
    --model-loader-extra-config "$BNB_CONFIG" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --trust-remote-code \
    --port "$PORT" \
    2>&1 | tee "$LOG_FILE" &