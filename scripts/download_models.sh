#!/usr/bin/env bash
# =============================================================================
# download_models.sh
#
# Downloads GGUF model files from HuggingFace into models/ using the
# huggingface_hub Python library (already in requirements.txt).
#
# Files are skipped if they already exist.
#
# Usage:
#   bash scripts/download_models.sh           # E4B pair (default)
#   bash scripts/download_models.sh --e2b     # E2B pair
#   bash scripts/download_models.sh --26b     # 26B A4B pair  (needs 32GB GPU for INT8)
#   bash scripts/download_models.sh --31b     # 31B pair      (needs 40GB+ GPU for INT8)
#
# After download, update TARGET_MODEL_GGUF and DRAFT_MODEL_GGUF in .env
# to point to the downloaded file paths printed at the end.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"
ENV_FILE="$REPO_ROOT/.env"

# Load HF_TOKEN from .env if present
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "[ERROR] HF_TOKEN is not set."
    echo "        Add it to .env:  HF_TOKEN=hf_your_token_here"
    exit 1
fi

mkdir -p "$MODELS_DIR"

# -----------------------------------------------------------------------
# Model selection
# -----------------------------------------------------------------------
MODEL_ARG="${1:---e4b}"

case "$MODEL_ARG" in
    --e2b)
        TARGET_REPO="ggml-org/gemma-4-E2B-it-GGUF"
        TARGET_FILE="gemma-4-E2B-it-Q8_0.gguf"
        DRAFT_REPO="AtomicChat/gemma-4-E2B-it-assistant-GGUF"
        DRAFT_FILE="gemma-4-E2B-it-assistant-F16.gguf"
        MODEL_ALIAS="gemma-4-e2b-it"
        ;;
    --26b)
        # Q4_K_M fits on 2x16GB V100 (split across both)
        TARGET_REPO="ggml-org/gemma-4-26B-A4B-it-GGUF"
        TARGET_FILE="gemma-4-26B-A4B-it-Q4_K_M.gguf"
        DRAFT_REPO="AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF"
        DRAFT_FILE="gemma-4-26B-A4B-it-assistant.Q4_K_M.gguf"
        MODEL_ALIAS="gemma-4-26b-a4b-it"
        ;;
    --31b)
        # Q4_K_M (~18GB) fits on 2x16GB V100 split
        TARGET_REPO="ggml-org/gemma-4-31B-it-GGUF"
        TARGET_FILE="gemma-4-31B-it-Q4_K_M.gguf"
        DRAFT_REPO="AtomicChat/gemma-4-31B-it-assistant-GGUF"
        DRAFT_FILE="gemma-4-31B-it-assistant-Q8_0.gguf"
        MODEL_ALIAS="gemma-4-31b-it"
        ;;
    --e4b)
        # Q8_0 (~4.3GB) — fits easily on a single 16GB V100
        TARGET_REPO="unsloth/gemma-4-E4B-it-GGUF"
        TARGET_FILE="gemma-4-E4B-it-Q8_0.gguf"
        DRAFT_REPO="AtomicChat/gemma-4-E4B-it-assistant-GGUF"
        DRAFT_FILE="gemma-4-E4B-it-assistant.Q8_0.gguf"
        MODEL_ALIAS="gemma-4-e4b-it"
        ;;
    --qwen3627b)
        # Q8_0 (~4.3GB) — fits easily on a single 16GB V100
        TARGET_REPO="unsloth/Qwen3.6-27B-MTP-GGUF"
        TARGET_FILE="Qwen3.6-27B-UD-Q4_K_XL.gguf"
        MODEL_ALIAS="qwen3.6-27b-mtp"
        ;;
    --qwen36-mmproj)
        TARGET_REPO="unsloth/Qwen3.6-27B-MTP-GGUF"
        TARGET_FILE="mmproj-F16.gguf"
        MODEL_ALIAS="qwen3.6-27b-mmproj"
        ;;
    --qwopus3627b|*)
        # Q8_0 (~4.3GB) — fits easily on a single 16GB V100
        TARGET_REPO="Jackrong/Qwopus3.6-27B-v2-MTP-GGUF"
        TARGET_FILE="Qwopus3.6-27B-v2-MTP-Q4_K_M.gguf"
        MODEL_ALIAS="qwopus-3.6-27b-v2-mtp"
        ;;
esac

echo "========================================================"
echo " Downloading GGUF models to $MODELS_DIR/"
echo " Target : $TARGET_FILE  (from $TARGET_REPO)"
# echo " Draft  : $DRAFT_FILE  (from $DRAFT_REPO)"
echo "========================================================"
echo ""

# -----------------------------------------------------------------------
# Download helper — skips if file already present
# -----------------------------------------------------------------------
download_file() {
    local repo="$1"
    local filename="$2"
    local dest="$MODELS_DIR/$filename"

    if [ -f "$dest" ]; then
        SIZE=$(du -h "$dest" | cut -f1)
        echo "[SKIP] $filename already exists ($SIZE)"
        return
    fi

    echo "Downloading $filename …"
    python3 - <<PYEOF
import os, sys
from huggingface_hub import hf_hub_download

try:
    path = hf_hub_download(
        repo_id="$repo",
        filename="$filename",
        local_dir="$MODELS_DIR",
        token=os.environ["HF_TOKEN"],
    )
    size_mb = os.path.getsize(path) / 1024**2
    print(f"  -> {path}  ({size_mb:.0f} MB)")
except Exception as e:
    print(f"[ERROR] Failed to download $filename: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

download_file "$TARGET_REPO" "$TARGET_FILE"
# download_file "$DRAFT_REPO"  "$DRAFT_FILE"

TARGET_PATH="$MODELS_DIR/$TARGET_FILE"
# DRAFT_PATH="$MODELS_DIR/$DRAFT_FILE"

echo ""
echo "========================================================"
echo " [OK] Models ready"
echo "========================================================"
ls -lh "$MODELS_DIR/"
echo ""
echo "Add these lines to your .env:"
echo ""
echo "  MODEL_ALIAS=$MODEL_ALIAS"
echo "  TARGET_MODEL_GGUF=$TARGET_PATH"
# echo "  DRAFT_MODEL_GGUF=$DRAFT_PATH"
echo "  TARGET_MODEL=$MODEL_ALIAS"
