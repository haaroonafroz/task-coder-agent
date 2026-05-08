#!/usr/bin/env bash
# =============================================================================
# stop_server.sh
#
# Gracefully stops a running vLLM server using the PID file written by
# the start scripts.
#
# Usage:
#   bash scripts/stop_server.sh baseline
#   bash scripts/stop_server.sh speculative
# =============================================================================

set -euo pipefail

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    echo "Usage: bash scripts/stop_server.sh [baseline|speculative]"
    exit 1
fi

PID_FILE="experiments/logs/${MODE}_server.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "[INFO] No PID file found at $PID_FILE. Server may not be running."
    exit 0
fi

PID=$(cat "$PID_FILE")
echo "Stopping $MODE server (PID $PID) …"

if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    sleep 3
    if kill -0 "$PID" 2>/dev/null; then
        echo "Process still alive; sending SIGKILL …"
        kill -9 "$PID"
    fi
    echo "[OK] Server stopped."
else
    echo "[INFO] Process $PID is not running."
fi

rm -f "$PID_FILE"

# Also free GPU memory by waiting for processes to release
echo "Waiting 10s for GPU memory to free …"
sleep 10
echo "Done."
