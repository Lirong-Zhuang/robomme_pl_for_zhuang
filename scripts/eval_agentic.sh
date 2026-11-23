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
# Hard wall-clock limit for each episode, including planning, grounding,
# execution, and verification correction calls.
MAX_EPISODE_SECONDS=600
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
# Used exactly as written; no model-name suffix is added automatically.
RUN_NAME="${RUN_NAME:-test_v1.7}"
# true deletes this run directory; false resumes non-error episode records and
# reuses an already generated task config.
OVERWRITE=false
USE_TASK_CONFIG_EXAMPLES=true

# =============================================================================
# Three-environment/model configuration
# =============================================================================

# Choose exactly one Agentic VLM: "30b" or "27b". This can also be overridden
# without editing the file, for example: VLM_MODEL=27b bash scripts/eval_agentic.sh
# Both models are served from the shared micromamba environment named "vllm".
VLM_MODEL="${VLM_MODEL:-30b}"
VLM_MODEL="${VLM_MODEL,,}"

VLM_MAMBA_ENV="vllm"
F2G_MAMBA_ENV="f2g"
EVAL_MAMBA_ENV="robomme"

MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/data/zhuanglr/micromamba}"
MAMBA_EXE="${MAMBA_EXE:-$MAMBA_ROOT_PREFIX/bin/micromamba}"

case "$VLM_MODEL" in
    30b)
        MODEL_PATH="/data/public/qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
        VLM_SERVED_MODEL_NAME="qwen3vl-30b"
        VLM_POINT_COORDINATE_SPACE="normalized_1000"
        VLM_EXTRA_ARGS=()
        ;;
    27b)
        MODEL_PATH="/data/public/qwen/Qwen3.8-27B-FP8"
        VLM_SERVED_MODEL_NAME="qwen38-27b"
        VLM_POINT_COORDINATE_SPACE="normalized_1000"
        VLM_EXTRA_ARGS=(
            --reasoning-parser qwen3
            --default-chat-template-kwargs '{"enable_thinking":false,"preserve_thinking":false}'
        )
        ;;
    *)
        echo "ERROR: VLM_MODEL must be '30b' or '27b', got: $VLM_MODEL" >&2
        exit 1
        ;;
esac

VLM_MAX_TOKENS=8192
VLM_MAX_MODEL_LEN=32768
VLM_GPU_MEMORY_UTILIZATION=0.90

F2G_ROOT="${F2G_ROOT:-$REPO_ROOT/../F2G-Pose}"
F2G_CHECKPOINT="${F2G_CHECKPOINT:-$F2G_ROOT/weights/f2g-pose.pt}"
F2G_RADIO_REPOSITORY="${F2G_RADIO_REPOSITORY:-$F2G_ROOT/third_party_local/RADIO}"
F2G_RADIO_CHECKPOINT="${F2G_RADIO_CHECKPOINT:-$F2G_ROOT/weights/radio-v2.5-l_half.pth.tar}"

CUDA_VISIBLE_DEVICES_AGENTIC="${CUDA_VISIBLE_DEVICES_AGENTIC:-0}"
VLM_HOST="127.0.0.1"
VLM_BIND_HOST="127.0.0.1"
VLM_PORT=0
F2G_HOST="127.0.0.1"
F2G_PORT=8765
VLM_STARTUP_TIMEOUT=600
F2G_STARTUP_TIMEOUT=300

SERVER_LOG_DIR="$SAVE_DIR/server_logs"
EVAL_PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/third_party/robomme_benchmark/src:$REPO_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
F2G_ENV_PREFIX="$MAMBA_ROOT_PREFIX/envs/$F2G_MAMBA_ENV"
CACHE_ROOT="${CACHE_ROOT:-/data/zhuanglr/cache}"

export MAMBA_ROOT_PREFIX
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_ROOT/vllm}"
export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-$CACHE_ROOT/flashinfer}"
export FLASHINFER_AUTOTUNE_CACHE_DIR="${FLASHINFER_AUTOTUNE_CACHE_DIR:-$FLASHINFER_CACHE_DIR/autotune}"
mkdir -p \
    "$HF_HOME" \
    "$UV_CACHE_DIR" \
    "$PIP_CACHE_DIR" \
    "$TORCH_HOME" \
    "$VLLM_CACHE_ROOT" \
    "$FLASHINFER_AUTOTUNE_CACHE_DIR"

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
# Resolve through a possible runs/ symlink so vLLM authorizes the same canonical
# /data path that pathlib uses for saved observation images.
RUN_MEDIA_ROOT=$(python3 -c \
    'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' \
    "$SAVE_DIR/$RUN_NAME")
mkdir -p "$RUN_MEDIA_ROOT"
if ((NUM_EPISODES < 1 || NUM_REPEATS < 1 || MAX_STEPS < 1 || MAX_DECISIONS < 1 || MAX_EPISODE_SECONDS < 1 || HISTORY_SIZE < 1 || HISTORY_STRIDE < 1 || MAX_INITIAL_OBSERVATIONS < 1 || PLANNER_ATTEMPTS < 1 || GROUNDING_CAMERA_RESOLUTION < 1)); then
    echo "ERROR: all episode, step, history, attempt, and resolution values must be positive." >&2
    exit 1
fi
for required_command in lsof curl; do
    if ! command -v "$required_command" &>/dev/null; then
        echo "ERROR: required command '$required_command' was not found." >&2
        exit 1
    fi
done
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

"$MAMBA_EXE" run -n "$VLM_MAMBA_ENV" python -c \
    'import torch, transformers, vllm; from packaging.version import Version; assert Version(transformers.__version__) >= Version("5.8.0")' \
    || {
        echo "ERROR: $VLM_MAMBA_ENV lacks vLLM dependencies." >&2
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
    'from f2g_pose import PoseEstimator; import numpy' \
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
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
"$MAMBA_EXE" run -n "$VLM_MAMBA_ENV" \
    python "$REPO_ROOT/scripts/snapshot_gpu_memory.py" \
    --cuda-visible-devices "$CUDA_VISIBLE_DEVICES_AGENTIC" \
    --output "$GPU_BASELINE_FILE"

args=(
    --vlm-host "$VLM_HOST"
    --vlm-port "$VLM_PORT"
    --vlm-model "$VLM_SERVED_MODEL_NAME"
    --vlm-model-path "$MODEL_PATH"
    --vlm-max-tokens "$VLM_MAX_TOKENS"
    --vlm-point-coordinate-space "$VLM_POINT_COORDINATE_SPACE"
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
    --max-episode-seconds "$MAX_EPISODE_SECONDS"
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
VLM_LOG="$SERVER_LOG_DIR/${RUN_NAME}_${VLM_SERVED_MODEL_NAME}_port${VLM_PORT}.log"
EVAL_LOG="$SERVER_LOG_DIR/${RUN_NAME}_eval.log"
F2G_PID=""
VLM_PID=""
STATUS_READER_PID=""
STATUS_DIR=""
STATUS_FIFO=""

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
    stop_process "$STATUS_READER_PID"
    if [[ -n "$STATUS_FIFO" ]]; then
        rm -f -- "$STATUS_FIFO"
    fi
    if [[ -n "$STATUS_DIR" ]]; then
        rmdir -- "$STATUS_DIR" 2>/dev/null || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

echo "Agentic tasks: $TASKS"
echo "Episodes: ${EPISODE_IDS:-0..$((NUM_EPISODES - 1))}"
echo "Repeats: $NUM_REPEATS"
echo "Per-episode wall-clock limit: ${MAX_EPISODE_SECONDS}s"
echo "Result directory: $SAVE_DIR/$RUN_NAME"
echo "vLLM local-media root: $RUN_MEDIA_ROOT"
echo "Environments: VLM=$VLM_MAMBA_ENV, F2G=$F2G_MAMBA_ENV, eval=$EVAL_MAMBA_ENV"
echo "Agentic VLM: $VLM_MODEL ($MODEL_PATH)"
echo "VLM point coordinates: $VLM_POINT_COORDINATE_SPACE"

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

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_AGENTIC" \
"$MAMBA_EXE" run -n "$VLM_MAMBA_ENV" \
    vllm serve "$MODEL_PATH" \
    --served-model-name "$VLM_SERVED_MODEL_NAME" \
    --host "$VLM_BIND_HOST" \
    --port "$VLM_PORT" \
    --tensor-parallel-size 1 \
    --max-model-len "$VLM_MAX_MODEL_LEN" \
    --max-num-seqs 1 \
    --gpu-memory-utilization "$VLM_GPU_MEMORY_UTILIZATION" \
    --allowed-local-media-path "$RUN_MEDIA_ROOT" \
    "${VLM_EXTRA_ARGS[@]}" \
    >"$VLM_LOG" 2>&1 &
VLM_PID=$!

echo "Starting $VLM_SERVED_MODEL_NAME with vLLM (PID $VLM_PID)..."
start_time=$SECONDS
until curl --fail --silent --max-time 2 \
    "http://$VLM_HOST:$VLM_PORT/health" >/dev/null 2>&1; do
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

served_models_json=$(curl --fail --silent --max-time 10 \
    "http://$VLM_HOST:$VLM_PORT/v1/models")
python3 -c \
    'import json, sys; expected=sys.argv[1]; payload=json.loads(sys.argv[2]); ids={item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}; assert expected in ids, f"expected served model {expected!r}, available={sorted(ids)}"' \
    "$VLM_SERVED_MODEL_NAME" "$served_models_json"

STATUS_DIR=$(mktemp -d)
STATUS_FIFO="$STATUS_DIR/agentic-status.fifo"
mkfifo "$STATUS_FIFO"
cat "$STATUS_FIFO" &
STATUS_READER_PID=$!

echo "Both services are ready. Starting RoboMME Agentic evaluation..."
if ! CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_AGENTIC" \
    PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$EVAL_PYTHONPATH" \
    AGENTIC_STATUS_PATH="$STATUS_FIFO" \
    "$MAMBA_EXE" run -n "$EVAL_MAMBA_ENV" \
        python "$REPO_ROOT/examples/robomme/eval_agentic.py" "${args[@]}" \
        >"$EVAL_LOG" 2>&1; then
    echo "ERROR: Agentic evaluation failed. Details: $EVAL_LOG" >&2
    exit 1
fi
