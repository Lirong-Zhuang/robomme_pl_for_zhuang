#!/usr/bin/env bash

set -Eeuo pipefail

# Run this script from the repository root:
#   bash scripts/eval.sh
#
# The Executer policy server is started as a managed background process because eval.py
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
EVAL_PRESET="symbolic_simpleSG_qwenvl"

EXECUTER_SEED=7
EXECUTER_CKPT_ID=79999
EXECUTER_GPU_ID=0
MANAGER_REPORTER_GPU_ID=0

# Set EXECUTER_PORT=0 to choose a free port automatically.
EXECUTER_HOST="0.0.0.0"
EXECUTER_PORT=0

# Task selection. Comma-separated values are supported.
# Counting suite: 4 tasks x 50 episodes = 200 episodes.
ONLY_TASKS="BinFill"
EXCLUDE_TASKS=""
RE_EVAL_TASKS=""

# Exact episode IDs override NUM_EPISODES. Examples: "4" or "2,7,17".
# Set EPISODE_IDS="" to evaluate episodes 0..NUM_EPISODES-1.
EPISODE_IDS=""
NUM_EPISODES=50

OBS_HORIZON=16
MAX_STEPS=1300
SUBGOAL_KEEP_PERIOD=1
SAVE_DIR="runs/evaluation"
# Optional final directory name for this evaluation run. When set, results are
# written under <SAVE_DIR>/<policy>/ckpt<id>/seed<seed>/<EVAL_RUN_NAME>/.
# Leave empty to use the Manager name (qwenvl, memer, gemini, or oracle).
EVAL_RUN_NAME="trinity_v0.1"
# Preserve completed tasks/episodes and continue with anything still missing.
OVERWRITE=true

# Save the per-episode Manager trace under
# <SAVE_DIR>/.../<TASK_NAME>/manager_logs/, while still displaying it in the
# terminal. Reporter requests have their own reporter_logs directory.
SAVE_MANAGER_LOGS=true

# Keep MemER's per-step images after an episode finishes. MemER keyframes are
# references to a subset of these images, so retaining keyframes requires
# retaining the episode image directory. This is a no-op for non-MemER models.
MANAGER_SAVE_MEMER_KF=true

# "auto" disables history only for pi05_baseline and enables it otherwise.
# It can also be set explicitly to "true" or "false".
EXECUTER_USE_HISTORY="auto"

# Manager configuration.
MANAGER_SIMPLE_ADAPTER_PATH="runs/ckpts/vlm_subgoal_predictor/qwenvl_baseline_v1.3_simple_subgoal/v0-20260812-185855/checkpoint-1300"
MANAGER_GROUNDED_ADAPTER_PATH="runs/ckpts/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200"

# Executer configuration
EXECUTER_CONFIG="mme_vla_suite"
EXECUTER_DIR="runs/ckpts/mme_vla_suite/symbolic-simple-subgoal/79999"

# Reporter configuration.
REPORTER_TYPE="qwenvl"
REPORTER_MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"

# micromamba
MAMBA_ENV="robomme"
MAMBA_ROOT_PREFIX="/data/zhuanglr/micromamba"
MAMBA_EXE="/data/zhuanglr/micromamba/bin/micromamba"

SERVER_STARTUP_TIMEOUT=180
SERVER_LOG_DIR="runs/evaluation/server_logs"

# Leave empty to use JAX's default. For a dedicated policy GPU, values such as
# 0.90 or 0.95 can be useful.
XLA_MEM_FRACTION="0.5"

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

# if [[ ! -f "$CONDA_INIT" ]]; then
#     echo "ERROR: conda initialization script not found: $CONDA_INIT" >&2
#     echo "Set CONDA_INIT near the top of scripts/eval.sh to the correct path." >&2
#     exit 1
# fi

REQUESTED_EVAL_PRESET=$EVAL_PRESET
EXECUTER_CONFIG_TYPE="mme_vla_suite"
EXECUTER_NAME=$EVAL_PRESET
MANAGER_TYPE="none"
SUBGOAL_TYPE="None"

case "$EVAL_PRESET" in
    pi05_baseline)
        EXECUTER_CONFIG_TYPE="pi05_baseline"
        EXECUTER_NAME="pi05_baseline"
        ;;
    MemER)
        EXECUTER_NAME="symbolic-grounded-subgoal"
        MANAGER_TYPE="memer"
        SUBGOAL_TYPE="grounded_subgoal"
        ;;
    symbolic_simpleSG_oracle)
        EXECUTER_NAME="symbolic-simple-subgoal"
        MANAGER_TYPE="oracle"
        SUBGOAL_TYPE="simple_subgoal"
        ;;
    symbolic_simpleSG_qwenvl)
        EXECUTER_NAME="symbolic-simple-subgoal"
        MANAGER_TYPE="qwenvl"
        SUBGOAL_TYPE="simple_subgoal"
        ;;
    symbolic_simpleSG_gemini)
        EXECUTER_NAME="symbolic-simple-subgoal"
        MANAGER_TYPE="gemini"
        SUBGOAL_TYPE="simple_subgoal"
        ;;
    symbolic_groundedSG_oracle)
        EXECUTER_NAME="symbolic-grounded-subgoal"
        MANAGER_TYPE="oracle"
        SUBGOAL_TYPE="grounded_subgoal"
        ;;
    symbolic_groundedSG_qwenvl)
        EXECUTER_NAME="symbolic-grounded-subgoal"
        MANAGER_TYPE="qwenvl"
        SUBGOAL_TYPE="grounded_subgoal"
        ;;
    symbolic_groundedSG_gemini)
        EXECUTER_NAME="symbolic-grounded-subgoal"
        MANAGER_TYPE="gemini"
        SUBGOAL_TYPE="grounded_subgoal"
        ;;
    perceptual-*|recurrent-*)
        EXECUTER_NAME=$EVAL_PRESET
        ;;
    *)
        echo "ERROR: unsupported EVAL_PRESET '$EVAL_PRESET'." >&2
        exit 1
        ;;
esac

if [[ -z "$EXECUTER_CONFIG" ]]; then
    EXECUTER_CONFIG=$EXECUTER_CONFIG_TYPE
fi
if [[ -z "$EXECUTER_DIR" ]]; then
    EXECUTER_DIR="runs/ckpts/$EXECUTER_CONFIG_TYPE/$EXECUTER_NAME/$EXECUTER_CKPT_ID"
fi
if [[ "$EXECUTER_PORT" == "0" ]]; then
    EXECUTER_PORT=$(find_free_port)
fi

if [[ "$EXECUTER_USE_HISTORY" == "auto" ]]; then
    if [[ "$REQUESTED_EVAL_PRESET" == "pi05_baseline" ]]; then
        EXECUTER_USE_HISTORY=false
    else
        EXECUTER_USE_HISTORY=true
    fi
fi

EVAL_ARGS=(
    --args.executer-host "$EXECUTER_HOST"
    --args.executer-port "$EXECUTER_PORT"
    --args.obs-horizon "$OBS_HORIZON"
    --args.max-steps "$MAX_STEPS"
    --args.save-dir "$SAVE_DIR"
    --args.run-name "$EVAL_RUN_NAME"
    --args.executer-name "$EXECUTER_NAME"
    --args.executer-seed "$EXECUTER_SEED"
    --args.executer-ckpt-id "$EXECUTER_CKPT_ID"
    --args.only-tasks "$ONLY_TASKS"
    --args.exclude-tasks "$EXCLUDE_TASKS"
    --args.re-eval-tasks "$RE_EVAL_TASKS"
    --args.num-episodes "$NUM_EPISODES"
    --args.episode-ids "$EPISODE_IDS"
    --args.subgoal-type "$SUBGOAL_TYPE"
    --args.subgoal-keep-period "$SUBGOAL_KEEP_PERIOD"
    --args.manager-simple-adapter-path "$MANAGER_SIMPLE_ADAPTER_PATH"
    --args.manager-grounded-adapter-path "$MANAGER_GROUNDED_ADAPTER_PATH"
    --args.reporter-type "$REPORTER_TYPE"
    --args.reporter-model-path "$REPORTER_MODEL_PATH"
)

EVAL_ARGS+=("$(bool_arg "$OVERWRITE" --args.overwrite --args.no-overwrite)")
EVAL_ARGS+=("$(bool_arg "$SAVE_MANAGER_LOGS" --args.save-manager-logs --args.no-save-manager-logs)")
EVAL_ARGS+=("$(bool_arg "$EXECUTER_USE_HISTORY" --args.executer-use-history --args.no-executer-use-history)")
EVAL_ARGS+=("$(bool_arg "$MANAGER_SAVE_MEMER_KF" --args.manager-save-memer-kf --args.no-manager-save-memer-kf)")

case "$MANAGER_TYPE" in
    oracle)
        EVAL_ARGS+=(--args.manager-use-oracle --args.no-manager-use-qwenvl --args.no-manager-use-memer --args.no-manager-use-gemini)
        ;;
    qwenvl)
        EVAL_ARGS+=(--args.no-manager-use-oracle --args.manager-use-qwenvl --args.no-manager-use-memer --args.no-manager-use-gemini)
        ;;
    memer)
        EVAL_ARGS+=(--args.no-manager-use-oracle --args.no-manager-use-qwenvl --args.manager-use-memer --args.no-manager-use-gemini)
        ;;
    gemini)
        EVAL_ARGS+=(--args.no-manager-use-oracle --args.no-manager-use-qwenvl --args.no-manager-use-memer --args.manager-use-gemini)
        ;;
    none)
        EVAL_ARGS+=(--args.no-manager-use-oracle --args.no-manager-use-qwenvl --args.no-manager-use-memer --args.no-manager-use-gemini)
        ;;
esac

mkdir -p "$SERVER_LOG_DIR"
SERVER_LOG="$SERVER_LOG_DIR/${EXECUTER_NAME}_ckpt${EXECUTER_CKPT_ID}_seed${EXECUTER_SEED}_port${EXECUTER_PORT}.log"
SERVER_PID=""

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo
        echo "Stopping Executer server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

echo "Evaluation:      $REQUESTED_EVAL_PRESET"
echo "Manager:         $MANAGER_TYPE"
echo "Executer:        $EXECUTER_NAME"
echo "Reporter:        $REPORTER_TYPE"
echo "Subgoal type:    $SUBGOAL_TYPE"
echo "Executer config: $EXECUTER_CONFIG"
echo "Executer ckpt:   $EXECUTER_DIR"
echo "Evaluation run:  ${EVAL_RUN_NAME:-default}"
echo "Task(s):         ${ONLY_TASKS:-all}"
echo "Episode ID(s):   ${EPISODE_IDS:-0..$((NUM_EPISODES - 1))}"
echo "Executer GPU:    $EXECUTER_GPU_ID"
echo "Manager/Reporter GPU: $MANAGER_REPORTER_GPU_ID"
echo "Executer endpoint: $EXECUTER_HOST:$EXECUTER_PORT"
echo "Server log:      $SERVER_LOG"
echo
echo "Starting Executer server..."

SERVER_ENV=(CUDA_VISIBLE_DEVICES="$EXECUTER_GPU_ID")
if [[ -n "$XLA_MEM_FRACTION" ]]; then
    SERVER_ENV+=(XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEM_FRACTION")
fi

env "${SERVER_ENV[@]}" uv run scripts/serve_policy.py \
    --seed "$EXECUTER_SEED" \
    --port "$EXECUTER_PORT" \
    --policy.dir "$EXECUTER_DIR" \
    --policy.config "$EXECUTER_CONFIG" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

START_TIME=$SECONDS
until lsof -iTCP:"$EXECUTER_PORT" -sTCP:LISTEN &>/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Executer server exited before it started listening." >&2
        tail -n 100 "$SERVER_LOG" >&2
        exit 1
    fi
    if ((SECONDS - START_TIME >= SERVER_STARTUP_TIMEOUT)); then
        echo "ERROR: Executer server did not start within ${SERVER_STARTUP_TIMEOUT}s." >&2
        tail -n 100 "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 1
done

echo "Executer server is ready. Starting evaluation in the foreground..."
echo "Press Ctrl+C to stop evaluation and the Executer server."
echo

# Launching the server first keeps it in the uv/openpi environment. The client
# then runs in the separate RoboMME conda environment in this foreground shell.
# source "$CONDA_INIT"
# conda activate "$CONDA_ENV"

printf 'Evaluation command:\n  CUDA_VISIBLE_DEVICES=%q python examples/robomme/eval.py' "$MANAGER_REPORTER_GPU_ID"
printf ' %q' "${EVAL_ARGS[@]}"
printf '\n\n'

CUDA_VISIBLE_DEVICES="$MANAGER_REPORTER_GPU_ID" \
MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX" \
"$MAMBA_EXE" run -n "$MAMBA_ENV" \
python examples/robomme/eval.py "${EVAL_ARGS[@]}"
