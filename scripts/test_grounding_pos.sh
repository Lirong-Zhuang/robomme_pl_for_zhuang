#!/usr/bin/env bash

# One-command launcher for the interactive F2G pose-viewer diagnostic.
# It starts the F2G service in the background, waits until it is ready, runs
# test_grounding_pos.py in the RoboMME environment, and always stops the service.

set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

EVAL_MAMBA_ENV="robomme"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/data/zhuanglr/micromamba}"
MAMBA_EXE="${MAMBA_EXE:-$MAMBA_ROOT_PREFIX/bin/micromamba}"
F2G_HOST="127.0.0.1"
F2G_PORT=8765
F2G_STARTUP_TIMEOUT="${F2G_STARTUP_TIMEOUT:-300}"
OUTPUT_DIR="$REPO_ROOT/runs/tests/grounding"
F2G_LOG="$OUTPUT_DIR/test_grounding_pos_f2g.log"
F2G_PID=""

stop_process() {
    local pid=$1
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    stop_process "$F2G_PID"
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

if [[ ! -x "$MAMBA_EXE" ]]; then
    echo "ERROR: micromamba is not executable: $MAMBA_EXE" >&2
    exit 1
fi
if ! command -v lsof &>/dev/null; then
    echo "ERROR: required command 'lsof' was not found." >&2
    exit 1
fi
if lsof -iTCP:"$F2G_PORT" -sTCP:LISTEN &>/dev/null; then
    echo "ERROR: port $F2G_PORT is already in use; stop the existing service first." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Starting the F2G service in the background..."
bash "$REPO_ROOT/scripts/serve_f2g_for_test.sh" >"$F2G_LOG" 2>&1 &
F2G_PID=$!

start_time=$SECONDS
until lsof -iTCP:"$F2G_PORT" -sTCP:LISTEN &>/dev/null; do
    if ! kill -0 "$F2G_PID" 2>/dev/null; then
        echo "ERROR: F2G service exited before listening." >&2
        tail -n 100 "$F2G_LOG" >&2
        exit 1
    fi
    if ((SECONDS - start_time >= F2G_STARTUP_TIMEOUT)); then
        echo "ERROR: F2G service startup timed out after ${F2G_STARTUP_TIMEOUT}s." >&2
        tail -n 100 "$F2G_LOG" >&2
        exit 1
    fi
    sleep 1
done

echo "F2G is ready at http://$F2G_HOST:$F2G_PORT"
echo "Running the interactive grounding-pose diagnostic..."

PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
"$MAMBA_EXE" run -n "$EVAL_MAMBA_ENV" \
    python "$REPO_ROOT/src/harness/tests/test_grounding_pos.py"

echo "Pose viewer generated successfully."
echo "F2G log: $F2G_LOG"
