#!/usr/bin/env bash

set -Eeuo pipefail

# Run this script from the repository root:
#   bash scripts/eval.sh
#
# The policy server is started as a managed background process because eval.py
# must run at the same time. eval.py itself runs in the foreground, so all
# evaluation output is visible in this terminal. The server log is written to
# SERVER_LOG_DIR and the server is stopped automatically when evaluation ends.

# =============================================================================
# Evaluation configuration
# =============================================================================

# Supported values:
#   pi05_baseline, MemER
#   symbolic_simpleSG_oracle, symbolic_simpleSG_qwenvl, symbolic_simpleSG_gemini
#   symbolic_groundedSG_oracle, symbolic_groundedSG_qwenvl, symbolic_groundedSG_gemini
#   perceptual-framesamp-context, perceptual-framesamp-modul, perceptual-framesamp-expert
#   perceptual-tokendrop-context, perceptual-tokendrop-modul, perceptual-tokendrop-expert
#   recurrent-rmt-context, recurrent-rmt-modul, recurrent-rmt-expert
#   recurrent-ttt-context, recurrent-ttt-modul, recurrent-ttt-expert
MODEL_TYPE="MemER"

SEED=7
CKPT_ID=79999
GPU_ID_SERVER=0
GPU_ID_CLIENT=0

# Set PORT=0 to choose a free port automatically.
HOST="0.0.0.0"
PORT=0

# Task selection. Comma-separated values are supported.
ONLY_TASKS="BinFill"
EXCLUDE_TASKS=""
RE_EVAL_TASKS=""

# Exact episode IDs override NUM_EPISODES. Examples: "4" or "2,7,17".
# Set EPISODE_IDS="" to evaluate episodes 0..NUM_EPISODES-1.
EPISODE_IDS="4"
NUM_EPISODES=2

OBS_HORIZON=16
MAX_STEPS=1300
SUBGOAL_KEEP_PERIOD=1
SAVE_DIR="runs/evaluation"
OVERWRITE=true

# Save stdout/stderr from each episode to a separate file under
# <SAVE_DIR>/.../episode_logs/, while still displaying it in the terminal.
SAVE_EPISODE_LOGS=true

# Keep MemER's per-step images after an episode finishes. MemER keyframes are
# references to a subset of these images, so retaining keyframes requires
# retaining the episode image directory. This is a no-op for non-MemER models.
SAVE_MEMER_KF=true

# "auto" disables history only for pi05_baseline and enables it otherwise.
# It can also be set explicitly to "true" or "false".
USE_HISTORY="auto"

# VLM configuration.
GEMINI_MODEL_NAME="gemini-2.5-pro"
QWENVL_SIMPLE_ADAPTER_PATH="runs/ckpts/vlm_subgoal_predictor/qwenvl/simple_subgoal/checkpoint-1400"
QWENVL_GROUNDED_ADAPTER_PATH="runs/ckpts/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200"
MEMER_ADAPTER_PATH="runs/ckpts/vlm_subgoal_predictor/memer/grounded_subgoal/checkpoint-1300"

# Runtime configuration.
CONDA_ENV="robomme"
# CONDA_INIT="$HOME/miniconda3/etc/profile.d/conda.sh"
CONDA_INIT="/opt/miniconda3/etc/profile.d/conda.sh"
SERVER_STARTUP_TIMEOUT=180
SERVER_LOG_DIR="runs/evaluation/server_logs"

# Leave empty to use JAX's default. For a dedicated policy GPU, values such as
# 0.90 or 0.95 can be useful.
XLA_MEM_FRACTION=""

# Optional overrides. Leave empty to use paths derived from MODEL_TYPE.
POLICY_DIR=""
POLICY_CONFIG=""

# =============================================================================
# Implementation
# =============================================================================

find_free_port() {
    local min=${1:-2000}
    local max=${2:-30000}
    local port
    local tries=5000

    for ((i = 0; i < tries; i++)); do
        port=$(shuf -i"${min}"-"${max}" -n1)
        if ! lsof -iTCP:"${port}" -sTCP:LISTEN &>/dev/null; then
            echo "${port}"
            return 0
        fi
    done

    echo "ERROR: no free port found in range ${min}-${max}" >&2
    return 1
}

bool_arg() {
    local value=$1
    local positive=$2
    local negative=$3
    case "$value" in
        true) printf '%s' "$positive" ;;
        false) printf '%s' "$negative" ;;
        *)
            echo "ERROR: expected true or false, got '$value'" >&2
            return 1
            ;;
    esac
}

for command_name in uv lsof shuf; do
    if ! command -v "$command_name" &>/dev/null; then
        echo "ERROR: required command '$command_name' was not found." >&2
        exit 1
    fi
done

if [[ ! -f "$CONDA_INIT" ]]; then
    echo "ERROR: conda initialization script not found: $CONDA_INIT" >&2
    echo "Set CONDA_INIT near the top of scripts/eval.sh to the correct path." >&2
    exit 1
fi

REQUESTED_MODEL_TYPE=$MODEL_TYPE
CONFIG_TYPE="mme_vla_suite"
POLICY_NAME=$MODEL_TYPE
PREDICTOR="none"
SUBGOAL_TYPE="None"

case "$MODEL_TYPE" in
    pi05_baseline)
        CONFIG_TYPE="pi05_baseline"
        POLICY_NAME="pi05_baseline"
        ;;
    MemER)
        POLICY_NAME="symbolic-grounded-subgoal"
        PREDICTOR="memer"
        SUBGOAL_TYPE="grounded_subgoal"
        ;;
    symbolic_simpleSG_oracle)
        POLICY_NAME="symbolic-simple-subgoal"
        PREDICTOR="oracle"
        SUBGOAL_TYPE="simple_subgoal"
        ;;
    symbolic_simpleSG_qwenvl)
        POLICY_NAME="symbolic-simple-subgoal"
        PREDICTOR="qwenvl"
        SUBGOAL_TYPE="simple_subgoal"
        ;;
    symbolic_simpleSG_gemini)
        POLICY_NAME="symbolic-simple-subgoal"
        PREDICTOR="gemini"
        SUBGOAL_TYPE="simple_subgoal"
        ;;
    symbolic_groundedSG_oracle)
        POLICY_NAME="symbolic-grounded-subgoal"
        PREDICTOR="oracle"
        SUBGOAL_TYPE="grounded_subgoal"
        ;;
    symbolic_groundedSG_qwenvl)
        POLICY_NAME="symbolic-grounded-subgoal"
        PREDICTOR="qwenvl"
        SUBGOAL_TYPE="grounded_subgoal"
        ;;
    symbolic_groundedSG_gemini)
        POLICY_NAME="symbolic-grounded-subgoal"
        PREDICTOR="gemini"
        SUBGOAL_TYPE="grounded_subgoal"
        ;;
    perceptual-*|recurrent-*)
        POLICY_NAME=$MODEL_TYPE
        ;;
    *)
        echo "ERROR: unsupported MODEL_TYPE '$MODEL_TYPE'." >&2
        exit 1
        ;;
esac

if [[ -z "$POLICY_CONFIG" ]]; then
    POLICY_CONFIG=$CONFIG_TYPE
fi
if [[ -z "$POLICY_DIR" ]]; then
    POLICY_DIR="runs/ckpts/$CONFIG_TYPE/$POLICY_NAME/$CKPT_ID"
fi
if [[ "$PORT" == "0" ]]; then
    PORT=$(find_free_port)
fi

if [[ "$USE_HISTORY" == "auto" ]]; then
    if [[ "$REQUESTED_MODEL_TYPE" == "pi05_baseline" ]]; then
        USE_HISTORY=false
    else
        USE_HISTORY=true
    fi
fi

EVAL_ARGS=(
    --args.host "$HOST"
    --args.port "$PORT"
    --args.obs-horizon "$OBS_HORIZON"
    --args.max-steps "$MAX_STEPS"
    --args.save-dir "$SAVE_DIR"
    --args.policy-name "$POLICY_NAME"
    --args.model-seed "$SEED"
    --args.model-ckpt-id "$CKPT_ID"
    --args.only-tasks "$ONLY_TASKS"
    --args.exclude-tasks "$EXCLUDE_TASKS"
    --args.re-eval-tasks "$RE_EVAL_TASKS"
    --args.num-episodes "$NUM_EPISODES"
    --args.episode-ids "$EPISODE_IDS"
    --args.subgoal-type "$SUBGOAL_TYPE"
    --args.subgoal-keep-period "$SUBGOAL_KEEP_PERIOD"
    --args.gemini-model-name "$GEMINI_MODEL_NAME"
    --args.qwenvl-simpleSG-adapter-path "$QWENVL_SIMPLE_ADAPTER_PATH"
    --args.qwenvl-groundSG-adapter-path "$QWENVL_GROUNDED_ADAPTER_PATH"
    --args.memer-adapter-path "$MEMER_ADAPTER_PATH"
)

EVAL_ARGS+=("$(bool_arg "$OVERWRITE" --args.overwrite --args.no-overwrite)")
EVAL_ARGS+=("$(bool_arg "$SAVE_EPISODE_LOGS" --args.save-episode-logs --args.no-save-episode-logs)")
EVAL_ARGS+=("$(bool_arg "$USE_HISTORY" --args.use-history --args.no-use-history)")
EVAL_ARGS+=("$(bool_arg "$SAVE_MEMER_KF" --args.save-memer-kf --args.no-save-memer-kf)")

case "$PREDICTOR" in
    oracle)
        EVAL_ARGS+=(--args.use-oracle --args.no-use-qwenvl --args.no-use-memer --args.no-use-gemini)
        ;;
    qwenvl)
        EVAL_ARGS+=(--args.no-use-oracle --args.use-qwenvl --args.no-use-memer --args.no-use-gemini)
        ;;
    memer)
        EVAL_ARGS+=(--args.no-use-oracle --args.no-use-qwenvl --args.use-memer --args.no-use-gemini)
        ;;
    gemini)
        EVAL_ARGS+=(--args.no-use-oracle --args.no-use-qwenvl --args.no-use-memer --args.use-gemini)
        ;;
    none)
        EVAL_ARGS+=(--args.no-use-oracle --args.no-use-qwenvl --args.no-use-memer --args.no-use-gemini)
        ;;
esac

mkdir -p "$SERVER_LOG_DIR"
SERVER_LOG="$SERVER_LOG_DIR/${POLICY_NAME}_ckpt${CKPT_ID}_seed${SEED}_port${PORT}.log"
SERVER_PID=""

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo
        echo "Stopping policy server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

echo "Model type:      $REQUESTED_MODEL_TYPE"
echo "Policy:          $POLICY_NAME"
echo "Predictor:       $PREDICTOR"
echo "Subgoal type:    $SUBGOAL_TYPE"
echo "Checkpoint:      $POLICY_DIR"
echo "Task(s):         ${ONLY_TASKS:-all}"
echo "Episode ID(s):   ${EPISODE_IDS:-0..$((NUM_EPISODES - 1))}"
echo "Server GPU:      $GPU_ID_SERVER"
echo "Client GPU:      $GPU_ID_CLIENT"
echo "Host/port:       $HOST:$PORT"
echo "Server log:      $SERVER_LOG"
echo
echo "Starting policy server..."

SERVER_ENV=(CUDA_VISIBLE_DEVICES="$GPU_ID_SERVER")
if [[ -n "$XLA_MEM_FRACTION" ]]; then
    SERVER_ENV+=(XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEM_FRACTION")
fi

env "${SERVER_ENV[@]}" uv run scripts/serve_policy.py \
    --seed "$SEED" \
    --port "$PORT" \
    --policy.dir "$POLICY_DIR" \
    --policy.config "$POLICY_CONFIG" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

START_TIME=$SECONDS
until lsof -iTCP:"$PORT" -sTCP:LISTEN &>/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: policy server exited before it started listening." >&2
        tail -n 100 "$SERVER_LOG" >&2
        exit 1
    fi
    if ((SECONDS - START_TIME >= SERVER_STARTUP_TIMEOUT)); then
        echo "ERROR: policy server did not start within ${SERVER_STARTUP_TIMEOUT}s." >&2
        tail -n 100 "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 1
done

echo "Policy server is ready. Starting evaluation in the foreground..."
echo "Press Ctrl+C to stop evaluation and the policy server."
echo

# Launching the server first keeps it in the uv/openpi environment. The client
# then runs in the separate RoboMME conda environment in this foreground shell.
source "$CONDA_INIT"
conda activate "$CONDA_ENV"

printf 'Evaluation command:\n  CUDA_VISIBLE_DEVICES=%q python examples/robomme/eval.py' "$GPU_ID_CLIENT"
printf ' %q' "${EVAL_ARGS[@]}"
printf '\n\n'

CUDA_VISIBLE_DEVICES="$GPU_ID_CLIENT" \
python examples/robomme/eval.py "${EVAL_ARGS[@]}"
