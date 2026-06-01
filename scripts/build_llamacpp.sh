#!/usr/bin/env bash
# =============================================================================
# build_llamacpp.sh
#
# Clones and builds the TurboQuant fork of llama.cpp from source.
# Must be built from source to compile CUDA kernels for V100 (sm_70).
# Pre-built binaries do not include sm_70 targets reliably.
#
# Targets: sm_70 (V100), sm_75 (T4/Turing), sm_80 (A100), sm_86 (RTX 3090/4090)
#
# Usage:
#   bash scripts/build_llamacpp.sh
#
# After build, binaries are in ~/llama-cpp-turboquant/build/bin/
# The script prints the exact export PATH line to add to your shell.
# =============================================================================

set -euo pipefail

REPO_URL="https://github.com/QuinsZouls/llama-cpp-turboquant.git"
BRANCH="llama-next"
BUILD_DIR="${LLAMACPP_BUILD_DIR:-$HOME/llama-cpp-turboquant}"

# sm_70 = V100, sm_75 = T4, sm_80 = A100, sm_86 = RTX 3090/A4000
CUDA_ARCHS="70;75;80;86"

echo "========================================================"
echo " Building TurboQuant llama.cpp"
echo " Branch: $BRANCH"
echo " CUDA arches: $CUDA_ARCHS (includes sm_70 for V100)"
echo " Build dir: $BUILD_DIR"
echo "========================================================"

# -----------------------------------------------------------------------
# 0. Prereqs check
# -----------------------------------------------------------------------
for cmd in cmake nvcc git; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "[ERROR] '$cmd' not found. Install it before running this script."
        case "$cmd" in
            cmake) echo "        apt-get install cmake" ;;
            nvcc)  echo "        CUDA toolkit must be installed (nvcc comes with it)." ;;
        esac
        exit 1
    fi
done

NVCC_VER=$(nvcc --version | grep "release" | sed 's/.*release //' | cut -d',' -f1)
echo "nvcc version: $NVCC_VER"
echo "cmake version: $(cmake --version | head -1)"
echo ""

# -----------------------------------------------------------------------
# 1. Clone or update
# -----------------------------------------------------------------------
if [ -d "$BUILD_DIR/.git" ]; then
    echo "[INFO] Repo already exists at $BUILD_DIR — pulling latest."
    cd "$BUILD_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo "Cloning $REPO_URL (branch: $BRANCH) …"
    git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$BUILD_DIR"
    cd "$BUILD_DIR"
fi

# -----------------------------------------------------------------------
# 2. Configure
# -----------------------------------------------------------------------
echo ""
echo "Configuring cmake …"
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF

# -----------------------------------------------------------------------
# 3. Build
# -----------------------------------------------------------------------
echo ""
echo "Building ($(nproc) cores) — this takes 5–15 minutes …"
cmake --build build --config Release -j"$(nproc)"

# -----------------------------------------------------------------------
# 4. Verify
# -----------------------------------------------------------------------
LLAMA_SERVER_BIN="$BUILD_DIR/build/bin/llama-server"
if [ ! -f "$LLAMA_SERVER_BIN" ]; then
    echo "[ERROR] Build failed — llama-server binary not found."
    exit 1
fi

echo ""
echo "========================================================"
echo " [OK] Build complete"
echo " Binary: $LLAMA_SERVER_BIN"
echo "========================================================"
echo ""
echo "Add to your shell session (or ~/.bashrc):"
echo ""
echo "  export PATH=\"$BUILD_DIR/build/bin:\$PATH\""
echo ""
echo "Verify:"
echo "  llama-server --version"
echo ""
echo "Quick smoke test (CPU only):"
echo "  llama-server --version && echo 'OK'"
