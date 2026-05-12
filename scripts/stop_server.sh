#!/usr/bin/env bash
# =============================================================================
# stop_server.sh
#
# Gracefully stops a running llama-server using the PID file written by
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

PORT=8000
[[ "$MODE" == "speculative" ]] && PORT=8001

PID_FILE="experiments/logs/${MODE}_server.pid"

# Kill by PID from file first
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "Stopping $MODE server (PID $PID) …"
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        sleep 2
    fi
    rm -f "$PID_FILE"
else
    echo "[INFO] No PID file at $PID_FILE."
fi

# Kill whatever is still holding the port — this is the reliable kill
PORT_PIDS=$(lsof -ti :"$PORT" 2>/dev/null || true)
if [[ -n "$PORT_PIDS" ]]; then
    echo "Killing process(es) still holding port $PORT: $PORT_PIDS"
    kill -9 $PORT_PIDS 2>/dev/null || true
fi

# Confirm
sleep 3
if lsof -Pi :"$PORT" -sTCP:LISTEN -t > /dev/null 2>&1; then
    echo "[ERROR] Port $PORT is still in use. Check manually: lsof -i :$PORT"
    exit 1
else
    echo "[OK] Port $PORT is free."
fi

echo "Waiting 5s for GPU memory to free …"
sleep 5
echo "Done."