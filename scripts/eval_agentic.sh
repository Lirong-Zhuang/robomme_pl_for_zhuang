#!/usr/bin/env bash

# Complete Agentic evaluation launcher.
#
# Edit the configuration block below, then run from any directory:
#   bash scripts/eval_agentic.sh

set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

# =============================================================================
# Evaluation configuration: task names, episodes, repeats, and run identity
# =============================================================================

# Comma-separated RoboMME task categories. Examples:
# BinFill, StopCube, PickXtimes, SwingXtimes, ButtonUnmask, VideoUnmask,
# VideoUnmaskSwap, ButtonUnmaskSwap, PickHighlight, VideoRepick,
# VideoPlaceButton, VideoPlaceOrder, MoveCube, InsertPeg, PatternLock, RouteStick
TASKS="BinFill"

# Exact comma-separated episode IDs override NUM_EPISODES. Leave empty to run
# episode IDs 0..NUM_EPISODES-1.
EPISODE_IDS=""
NUM_EPISODES=1

# Repeat the same benchmark episode IDs with the same metadata seed/difficulty.
NUM_REPEATS=1

MAX_STEPS=3000
MAX_DECISIONS=50
HISTORY_SIZE=3
HISTORY_STRIDE=16
MAX_INITIAL_OBSERVATIONS=16
PLANNER_ATTEMPTS=3
DEPTH_SCALE=0.001

# Native square resolution used by the independent grounding camera and F2G.
# The VLM/policy camera remains unchanged at its task-defined resolution.
# Change this value to 256, 512, 768, 1024, etc. before launching eval.
GROUNDING_CAMERA_RESOLUTION="${GROUNDING_CAMERA_RESOLUTION:-768}"

SAVE_DIR="runs/evaluation/agentic_v1"
RUN_NAME="test_v1.4"
# true deletes this run directory; false resumes non-error episode records and
# reuses an already generated task config.
OVERWRITE=false
USE_TASK_CONFIG_EXAMPLES=true

# =============================================================================
# Three-environment/model configuration
# =============================================================================

VLM_MAMBA_ENV="robomme-qwen38"
F2G_MAMBA_ENV="f2g"
EVAL_MAMBA_ENV="robomme"

MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/data/zhuanglr/micromamba}"
MAMBA_EXE="${MAMBA_EXE:-$MAMBA_ROOT_PREFIX/bin/micromamba}"

MODEL_PATH="${QWEN_MODEL_PATH:-/data/zhuanglr/models/Qwen3.8-27B-FP8-fixed}"
ATTN_IMPL="flash_attention_2"
VLM_MAX_TOKENS=8192
IMAGE_MAX_TOKEN_NUM=256

F2G_ROOT="${F2G_ROOT:-$REPO_ROOT/../F2G-Pose}"
F2G_CHECKPOINT="${F2G_CHECKPOINT:-$F2G_ROOT/weights/f2g-pose.pt}"
F2G_RADIO_REPOSITORY="${F2G_RADIO_REPOSITORY:-$F2G_ROOT/third_party_local/RADIO}"
F2G_RADIO_CHECKPOINT="${F2G_RADIO_CHECKPOINT:-$F2G_ROOT/weights/radio-v2.5-l_half.pth.tar}"

CUDA_VISIBLE_DEVICES_AGENTIC="${CUDA_VISIBLE_DEVICES_AGENTIC:-0}"
VLM_HOST="127.0.0.1"
VLM_BIND_HOST="0.0.0.0"
VLM_PORT=0
F2G_HOST="127.0.0.1"
F2G_PORT=8765
VLM_STARTUP_TIMEOUT=600
F2G_STARTUP_TIMEOUT=300

SERVER_LOG_DIR="$SAVE_DIR/server_logs"
VLM_PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
EVAL_PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/third_party/robomme_benchmark/src:$REPO_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
F2G_ENV_PREFIX="$MAMBA_ROOT_PREFIX/envs/$F2G_MAMBA_ENV"

export MAMBA_ROOT_PREFIX
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"
export IMAGE_MAX_TOKEN_NUM

# =============================================================================
# Validation and argument assembly
# =============================================================================

if [[ ! -x "$MAMBA_EXE" ]]; then
    echo "ERROR: micromamba is not executable: $MAMBA_EXE" >&2
    exit 1
fi
for required_path in "$MODEL_PATH" "$F2G_RADIO_REPOSITORY"; do
    if [[ ! -d "$required_path" ]]; then
        echo "ERROR: required directory does not exist: $required_path" >&2
        exit 1
    fi
done
for required_file in "$F2G_CHECKPOINT" "$F2G_RADIO_CHECKPOINT"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required file does not exist: $required_file" >&2
        exit 1
    fi
done
if [[ -z "$TASKS" || -z "$RUN_NAME" ]]; then
    echo "ERROR: TASKS and RUN_NAME must not be empty." >&2
    exit 1
fi
if ((NUM_EPISODES < 1 || NUM_REPEATS < 1 || MAX_DECISIONS < 1 || GROUNDING_CAMERA_RESOLUTION < 1)); then
    echo "ERROR: episode/repeat/decision counts and grounding resolution must be positive." >&2
    exit 1
fi
if ! command -v lsof &>/dev/null; then
    echo "ERROR: required command 'lsof' was not found." >&2
    exit 1
fi
if ((VLM_PORT == 0)); then
    VLM_PORT=$(python3 -c \
        'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
fi
for port in "$VLM_PORT" "$F2G_PORT"; do
    if lsof -iTCP:"$port" -sTCP:LISTEN &>/dev/null; then
        echo "ERROR: required port $port is already in use." >&2
        exit 1
    fi
done

PYTHONPATH="$VLM_PYTHONPATH" \
"$MAMBA_EXE" run -n "$VLM_MAMBA_ENV" python -c \
    'import msgpack, numpy, swift, torch, transformers, websockets' \
    || {
        echo "ERROR: $VLM_MAMBA_ENV lacks an Agentic VLM dependency." >&2
        exit 1
    }

PYTHONPATH="$EVAL_PYTHONPATH" \
"$MAMBA_EXE" run -n "$EVAL_MAMBA_ENV" python -c \
    'import cv2, gymnasium, imageio, mani_skill, mplib, msgpack, sapien, websockets' \
    || {
        echo "ERROR: $EVAL_MAMBA_ENV lacks a RoboMME evaluation dependency." >&2
        exit 1
    }

CUDA_HOME="$F2G_ENV_PREFIX" \
PATH="$F2G_ENV_PREFIX/bin:$PATH" \
PYTHONPATH="${PYTHONPATH:-}" \
"$MAMBA_EXE" run -n "$F2G_MAMBA_ENV" python -c \
    'from f2g_pose import PoseEstimator; import numpy; print("F2G import OK")' \
    || {
        echo "ERROR: $F2G_MAMBA_ENV cannot import F2G-Pose." >&2
        exit 1
    }

case "$OVERWRITE" in
    true) overwrite_arg="--overwrite" ;;
    false) overwrite_arg="--no-overwrite" ;;
    *) echo "ERROR: OVERWRITE must be true or false." >&2; exit 1 ;;
esac
case "$USE_TASK_CONFIG_EXAMPLES" in
    true) examples_arg="--use-task-config-examples" ;;
    false) examples_arg="--no-use-task-config-examples" ;;
    *) echo "ERROR: USE_TASK_CONFIG_EXAMPLES must be true or false." >&2; exit 1 ;;
esac

# Capture the card state before either process belonging to this evaluation is
# started. This keeps F2G out of the "other programs before eval" baseline.
mkdir -p "$SERVER_LOG_DIR"
GPU_BASELINE_FILE="$SERVER_LOG_DIR/${RUN_NAME}_pre_eval_gpu.json"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_AGENTIC" \
PYTHONPATH="$VLM_PYTHONPATH" \
"$MAMBA_EXE" run -n "$VLM_MAMBA_ENV" \
    python "$REPO_ROOT/scripts/snapshot_gpu_memory.py" \
    --cuda-visible-devices "$CUDA_VISIBLE_DEVICES_AGENTIC" \
    --output "$GPU_BASELINE_FILE"

args=(
    --vlm-host "$VLM_HOST"
    --vlm-port "$VLM_PORT"
    --f2g-url "http://$F2G_HOST:$F2G_PORT"
    --pre-eval-gpu-snapshot "$GPU_BASELINE_FILE"
    --depth-scale "$DEPTH_SCALE"
    --grounding-camera-resolution "$GROUNDING_CAMERA_RESOLUTION"
    --tasks "$TASKS"
    --episode-ids "$EPISODE_IDS"
    --num-episodes "$NUM_EPISODES"
    --num-repeats "$NUM_REPEATS"
    --max-steps "$MAX_STEPS"
    --max-decisions "$MAX_DECISIONS"
    --history-size "$HISTORY_SIZE"
    --history-stride "$HISTORY_STRIDE"
    --max-initial-observations "$MAX_INITIAL_OBSERVATIONS"
    --planner-attempts "$PLANNER_ATTEMPTS"
    --save-dir "$SAVE_DIR"
    --run-name "$RUN_NAME"
    "$overwrite_arg"
    "$examples_arg"
)

# =============================================================================
# Managed F2G and Qwen services
# =============================================================================

mkdir -p "$SERVER_LOG_DIR"
F2G_LOG="$SERVER_LOG_DIR/${RUN_NAME}_f2g_port${F2G_PORT}.log"
VLM_LOG="$SERVER_LOG_DIR/${RUN_NAME}_vlm_port${VLM_PORT}.log"
F2G_PID=""
VLM_PID=""

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
    stop_process "$VLM_PID"
    stop_process "$F2G_PID"
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

echo "Agentic tasks: $TASKS"
echo "Episodes: ${EPISODE_IDS:-0..$((NUM_EPISODES - 1))}"
echo "Repeats: $NUM_REPEATS"
echo "Result directory: $SAVE_DIR/$RUN_NAME"
echo "Environments: VLM=$VLM_MAMBA_ENV, F2G=$F2G_MAMBA_ENV, eval=$EVAL_MAMBA_ENV"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_AGENTIC" \
CUDA_HOME="$F2G_ENV_PREFIX" \
PATH="$F2G_ENV_PREFIX/bin:$PATH" \
LD_LIBRARY_PATH="$F2G_ENV_PREFIX/lib:$F2G_ENV_PREFIX/lib64:${LD_LIBRARY_PATH:-}" \
F2G_RADIO_REPOSITORY="$F2G_RADIO_REPOSITORY" \
F2G_RADIO_CHECKPOINT="$F2G_RADIO_CHECKPOINT" \
"$MAMBA_EXE" run -n "$F2G_MAMBA_ENV" \
    python "$REPO_ROOT/scripts/f2g_pose_server.py" \
    --host "$F2G_HOST" \
    --port "$F2G_PORT" \
    --checkpoint "$F2G_CHECKPOINT" \
    >"$F2G_LOG" 2>&1 &
F2G_PID=$!

echo "Starting F2G service (PID $F2G_PID)..."
start_time=$SECONDS
until lsof -iTCP:"$F2G_PORT" -sTCP:LISTEN &>/dev/null; do
    if ! kill -0 "$F2G_PID" 2>/dev/null; then
        echo "ERROR: F2G service exited before listening." >&2
        tail -n 100 "$F2G_LOG" >&2
        exit 1
    fi
    if ((SECONDS - start_time >= F2G_STARTUP_TIMEOUT)); then
        echo "ERROR: F2G service startup timed out." >&2
        tail -n 100 "$F2G_LOG" >&2
        exit 1
    fi
    sleep 1
done

export USE_HF=1
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_AGENTIC" \
PYTHONPATH="$VLM_PYTHONPATH" \
"$MAMBA_EXE" run -n "$VLM_MAMBA_ENV" \
    python "$REPO_ROOT/scripts/serve_agentic_vlm.py" \
    --host "$VLM_BIND_HOST" \
    --port "$VLM_PORT" \
    --model-path "$MODEL_PATH" \
    --attn-impl "$ATTN_IMPL" \
    --max-tokens "$VLM_MAX_TOKENS" \
    >"$VLM_LOG" 2>&1 &
VLM_PID=$!

echo "Starting Agentic Qwen service (PID $VLM_PID)..."
start_time=$SECONDS
until lsof -iTCP:"$VLM_PORT" -sTCP:LISTEN &>/dev/null; do
    if ! kill -0 "$VLM_PID" 2>/dev/null; then
        echo "ERROR: Agentic Qwen service exited before listening." >&2
        tail -n 100 "$VLM_LOG" >&2
        exit 1
    fi
    if ((SECONDS - start_time >= VLM_STARTUP_TIMEOUT)); then
        echo "ERROR: Agentic Qwen service startup timed out." >&2
        tail -n 100 "$VLM_LOG" >&2
        exit 1
    fi
    sleep 1
done

echo "Both services are ready. Starting RoboMME Agentic evaluation..."
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_AGENTIC" \
PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
PYTHONPATH="$EVAL_PYTHONPATH" \
"$MAMBA_EXE" run -n "$EVAL_MAMBA_ENV" \
    python "$REPO_ROOT/examples/robomme/eval_agentic.py" "${args[@]}"
