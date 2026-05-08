#!/usr/bin/env bash
# =============================================================================
# install_vllm.sh
#
# Sets up the environment for the speculative decoding evaluation.
# Run this ONCE before starting any experiments.
#
# Sections 0.1, 0.2, 0.3, 6.1 of the implementation plan.
# =============================================================================

set -euo pipefail

VLLM_PR_BRANCH="lucianommartins/gemma4-mtp"
VLLM_REPO="https://github.com/lucianommartins/vllm.git"
VLLM_PR_URL="https://github.com/vllm-project/vllm/pull/41745"

echo "========================================================"
echo " Speculative Decoding Eval – Environment Setup"
echo "========================================================"

# -----------------------------------------------------------------------
# Step 0: Check whether the PR has merged into vllm main
# -----------------------------------------------------------------------
echo ""
echo "[Step 0] Checking vLLM PR status …"
echo "  PR: $VLLM_PR_URL"
echo "  If this PR has merged by the time you run this script,"
echo "  install from main instead:  pip install vllm"
echo "  Otherwise, this script installs from the PR branch."
echo ""

read -r -p "Has PR #41745 merged into vllm main? [y/N] " MERGED
if [[ "${MERGED,,}" == "y" ]]; then
    echo "[INFO] Installing vLLM from PyPI (PR merged) …"
    pip install vllm
else
    echo "[INFO] Installing vLLM from PR branch …"
    if [ -d "vllm_pr" ]; then
        echo "  Existing vllm_pr/ found. Pulling latest …"
        git -C vllm_pr pull
    else
        git clone "$VLLM_REPO" -b "$VLLM_PR_BRANCH" vllm_pr
    fi
    pip install -e vllm_pr
fi

# -----------------------------------------------------------------------
# Step 1: Transformers from git main (requires gemma4_assistant type)
# Section 0.2
# -----------------------------------------------------------------------
echo ""
echo "[Step 1] Installing transformers from git main …"
echo "  (Required for gemma4_assistant model type)"
pip install git+https://github.com/huggingface/transformers.git

echo ""
echo "[Step 1] Verifying transformers import …"
python -c "from transformers import AutoModelForCausalLM; print('  transformers OK')"

# -----------------------------------------------------------------------
# Step 2: Remaining Python dependencies
# -----------------------------------------------------------------------
echo ""
echo "[Step 2] Installing remaining dependencies …"
pip install \
    langgraph>=0.2.0 \
    openai>=1.30.0 \
    langchain-core>=0.2.0 \
    pandas>=2.2.0 \
    numpy>=1.26.0 \
    scipy>=1.13.0 \
    matplotlib>=3.9.0 \
    huggingface_hub>=0.23.0 \
    accelerate>=0.30.0 \
    bitsandbytes>=0.43.0 \
    tqdm>=4.66.0 \
    python-dotenv>=1.0.0

# -----------------------------------------------------------------------
# Step 3: HuggingFace login (Section 0.3)
# -----------------------------------------------------------------------
echo ""
echo "[Step 3] HuggingFace login"
echo "  You must accept the Gemma 4 license at:"
echo "  https://huggingface.co/google/gemma-4-E4B-it"
echo ""

# Source .env if present so HF_TOKEN is available non-interactively.
if [ -f ".env" ]; then
    set -a
    # shellcheck source=/dev/null
    source .env
    set +a
    echo "  Loaded .env"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "  HF_TOKEN found in environment — logging in non-interactively."
    huggingface-cli login --token "$HF_TOKEN"
else
    echo "  No HF_TOKEN set. Falling back to interactive login."
    echo "  Tip: copy .env.example to .env and set HF_TOKEN to skip this prompt."
    huggingface-cli login
fi

# -----------------------------------------------------------------------
# Step 4: Prepare dataset
# -----------------------------------------------------------------------
echo ""
echo "[Step 4] Preparing frozen dataset …"
python datasets/prepare_dataset.py

echo ""
echo "========================================================"
echo " Setup complete."
echo " Next steps:"
echo "   1. Start baseline server:    bash scripts/start_server_baseline.sh"
echo "   2. Run calibration:          python run_experiment.py --mode baseline --calibration"
echo "   3. Run full baseline:        python run_experiment.py --mode baseline"
echo "   4. Stop baseline server, then start speculative:"
echo "                                bash scripts/start_server_speculative.sh"
echo "   5. Run full speculative:     python run_experiment.py --mode speculative"
echo "   6. Verify outputs:           python verify_outputs.py"
echo "   7. Analyze & chart:          python analysis/analyze_results.py"
echo "========================================================"
