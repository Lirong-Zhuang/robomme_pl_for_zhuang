#!/usr/bin/env bash

# Standalone F2G service for src/harness/tests/test_grounding.py.
# Edit this configuration block if the server paths differ.

set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
F2G_MAMBA_ENV="f2g"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/data/zhuanglr/micromamba}"
MAMBA_EXE="${MAMBA_EXE:-$MAMBA_ROOT_PREFIX/bin/micromamba}"
F2G_ENV_PREFIX="$MAMBA_ROOT_PREFIX/envs/$F2G_MAMBA_ENV"
F2G_ROOT="${F2G_ROOT:-$REPO_ROOT/../F2G-Pose}"
F2G_CHECKPOINT="${F2G_CHECKPOINT:-$F2G_ROOT/weights/f2g-pose.pt}"
F2G_RADIO_REPOSITORY="${F2G_RADIO_REPOSITORY:-$F2G_ROOT/third_party_local/RADIO}"
F2G_RADIO_CHECKPOINT="${F2G_RADIO_CHECKPOINT:-$F2G_ROOT/weights/radio-v2.5-l_half.pth.tar}"
F2G_HOST="127.0.0.1"
F2G_PORT=8765
CUDA_VISIBLE_DEVICES_F2G="${CUDA_VISIBLE_DEVICES_F2G:-0}"

for required_path in \
    "$MAMBA_EXE" \
    "$F2G_RADIO_REPOSITORY" \
    "$F2G_CHECKPOINT" \
    "$F2G_RADIO_CHECKPOINT"; do
    if [[ ! -e "$required_path" ]]; then
        echo "ERROR: required F2G path does not exist: $required_path" >&2
        echo "Edit the configuration block in scripts/serve_f2g_for_test.sh." >&2
        exit 1
    fi
done

export MAMBA_ROOT_PREFIX
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"

echo "Starting F2G test service at http://$F2G_HOST:$F2G_PORT"
echo "Checkpoint: $F2G_CHECKPOINT"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_F2G" \
CUDA_HOME="$F2G_ENV_PREFIX" \
PATH="$F2G_ENV_PREFIX/bin:$PATH" \
LD_LIBRARY_PATH="$F2G_ENV_PREFIX/lib:$F2G_ENV_PREFIX/lib64:${LD_LIBRARY_PATH:-}" \
F2G_RADIO_REPOSITORY="$F2G_RADIO_REPOSITORY" \
F2G_RADIO_CHECKPOINT="$F2G_RADIO_CHECKPOINT" \
exec "$MAMBA_EXE" run -n "$F2G_MAMBA_ENV" \
    python "$REPO_ROOT/scripts/f2g_pose_server.py" \
    --host "$F2G_HOST" \
    --port "$F2G_PORT" \
    --checkpoint "$F2G_CHECKPOINT"
