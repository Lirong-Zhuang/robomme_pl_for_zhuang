#!/usr/bin/env bash

set -Eeuo pipefail

# =============================================================================
# User configuration
# =============================================================================

# Checkpoints are written to:
# runs/ckpts/pi05_baseline_lora/${MODEL_NAME}/
MODEL_NAME="vla_baseline_v2"

DATASET_PATH="data/robomme_preprocessed_dup_binfill_data"
NUM_TRAIN_STEPS=30000

# BATCH_SIZE is the global batch. It must be divisible by the number of
# comma-separated GPU IDs; the script calculates the batch assigned per GPU.
GPU_IDS="0"
BATCH_SIZE=64

# Same checkpoint cadence as the original pi05_baseline configuration.
SAVE_INTERVAL=2000
KEEP_PERIOD=2000

# Set to true only after exporting a valid WANDB_API_KEY.
USE_WANDB=false

# =============================================================================
# Validation and derived training values
# =============================================================================

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

if [[ ! "$MODEL_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: MODEL_NAME may contain only letters, numbers, '.', '_' and '-'." >&2
    exit 1
fi
if [[ ! "$NUM_TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NUM_TRAIN_STEPS must be a positive integer." >&2
    exit 1
fi
if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: BATCH_SIZE must be a positive integer." >&2
    exit 1
fi

IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
GPU_COUNT=${#GPU_ID_ARRAY[@]}
if ((GPU_COUNT < 1)); then
    echo "ERROR: GPU_IDS must contain at least one GPU ID." >&2
    exit 1
fi

if ((BATCH_SIZE % GPU_COUNT != 0)); then
    echo "ERROR: global BATCH_SIZE ($BATCH_SIZE) must be divisible by GPU count ($GPU_COUNT)." >&2
    exit 1
fi
GLOBAL_BATCH_SIZE=$BATCH_SIZE
PER_GPU_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / GPU_COUNT))
STATS_PATH="$DATASET_PATH/meta/stats.json"
if [[ ! -f "$STATS_PATH" ]]; then
    echo "ERROR: dataset metadata was not found: $STATS_PATH" >&2
    exit 1
fi

DATASET_SIZE=$(
    uv run python -c \
        'import json, pathlib, sys; stats=json.loads(pathlib.Path(sys.argv[1]).read_text()); print(stats["execution_samples"] if "execution_samples" in stats else stats["total_samples"])' \
        "$STATS_PATH"
)
if [[ ! "$DATASET_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: invalid dataset size in $STATS_PATH: $DATASET_SIZE" >&2
    exit 1
fi

if ((DATASET_SIZE < GLOBAL_BATCH_SIZE)); then
    echo "ERROR: global batch size ($GLOBAL_BATCH_SIZE) exceeds dataset size ($DATASET_SIZE)." >&2
    exit 1
fi

NORM_STATS_PATH="runs/assets/pi05_baseline/robomme/norm_stats.json"
if [[ ! -f "$NORM_STATS_PATH" ]]; then
    echo "ERROR: normalization statistics were not found: $NORM_STATS_PATH" >&2
    echo "Run scripts/compute_norm_stats.py for config pi05_baseline first." >&2
    exit 1
fi

CHECKPOINT_DIR="runs/ckpts/pi05_baseline_lora/$MODEL_NAME"
if [[ -e "$CHECKPOINT_DIR" ]]; then
    echo "ERROR: checkpoint directory already exists: $CHECKPOINT_DIR" >&2
    echo "Choose a new MODEL_NAME to avoid overwriting an existing run." >&2
    exit 1
fi

WANDB_ARGS=(--no-wandb-enabled)
if [[ "$USE_WANDB" == "true" ]]; then
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        echo "ERROR: USE_WANDB=true but WANDB_API_KEY is not set." >&2
        exit 1
    fi
    WANDB_ARGS=(--wandb-enabled)
elif [[ "$USE_WANDB" != "false" ]]; then
    echo "ERROR: USE_WANDB must be true or false." >&2
    exit 1
fi

echo "Model name:          $MODEL_NAME"
echo "Dataset:             $DATASET_PATH"
echo "Dataset samples:     $DATASET_SIZE"
echo "Visible GPUs:        $GPU_IDS ($GPU_COUNT GPUs)"
echo "Batch per GPU:       $PER_GPU_BATCH_SIZE"
echo "Global batch size:   $GLOBAL_BATCH_SIZE"
echo "Total train steps:   $NUM_TRAIN_STEPS"
echo "Checkpoint dir:      $CHECKPOINT_DIR"

# Do not set XLA_PYTHON_CLIENT_PREALLOCATE, XLA_PYTHON_CLIENT_ALLOCATOR, or
# XLA_PYTHON_CLIENT_MEM_FRACTION here. JAX still necessarily uses XLA as its
# compiler/runtime, but this launcher does not request XLA memory preallocation
# or a virtual/unified-memory configuration. num-workers=0 also avoids spawning
# data-loader workers that set XLA allocator environment variables.
CUDA_VISIBLE_DEVICES="$GPU_IDS" \
uv run scripts/train.py pi05_baseline_lora \
    --exp-name="$MODEL_NAME" \
    --batch-size="$GLOBAL_BATCH_SIZE" \
    --num-workers=0 \
    --fsdp-devices="$GPU_COUNT" \
    --dataset-path="$DATASET_PATH" \
    --num-train-steps="$NUM_TRAIN_STEPS" \
    --save-interval="$SAVE_INTERVAL" \
    --keep-period="$KEEP_PERIOD" \
    --overwrite \
    "${WANDB_ARGS[@]}"
