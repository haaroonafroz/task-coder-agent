#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen3.6-35B-A3B"  # Original model
NUM_SPEC_TOKENS=3
PORT=8002
GPU_MEM_UTIL=0.90
MAX_MODEL_LEN=32768
LOG_FILE="experiments/logs/speculative_server.log"

mkdir -p experiments/logs

echo "========================================================"
echo " Starting Qwen 3.6 SPECULATIVE (bitsandbytes 4-bit)"
echo " Model:  $MODEL"
echo " Port:   $PORT"
echo "========================================================"

SPEC_CONFIG="{\"method\":\"mtp\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS}"
BNB_CONFIG='{"load_in_4bit":true,"bnb_4bit_compute_dtype":"bfloat16","bnb_4bit_quant_type":"nf4","bnb_4bit_use_double_quant":true}'

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --speculative-config "$SPEC_CONFIG" \
    --dtype bfloat16 \
    --quantization bitsandbytes \
    --model-loader-extra-config "$BNB_CONFIG" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --host 0.0.0.0 \
    --port "$PORT" \
    2>&1 | tee "$LOG_FILE" &