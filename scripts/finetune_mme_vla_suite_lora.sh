#!/usr/bin/env bash

set -Eeuo pipefail

# LoRA fine-tuning follows dev_trinity_excuter_v3: the 2B VLM expert uses
# LoRA, while the 300M action expert and action projections remain trainable.
MME_VLA_TYPE="symbolic-simple-subgoal"
TRAIN_CONFIG="mme_vla_suite_lora"
RUN_NAME="${MME_VLA_TYPE}_baseline_v0_lora"

# Keep these aligned with the full-parameter training run.
BATCH_SIZE=64
DATASET_PATH="data/robomme_preprocessed_dup_binfill_data"

# Single-GPU setup. FSDP_DEVICES must equal the number of visible GPUs.
GPU_IDS="0"
FSDP_DEVICES=1

# export WANDB_API_KEY=<YOUR_WANDB_API_KEY>

CUDA_VISIBLE_DEVICES="$GPU_IDS" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
uv run scripts/train.py "$TRAIN_CONFIG" \
    --exp-name="$RUN_NAME" \
    --batch-size="$BATCH_SIZE" \
    --num-workers=4 \
    --resume \
    --fsdp-devices="$FSDP_DEVICES" \
    --dataset-path="$DATASET_PATH" \
    --model.use-history \
    --model.history-config="${MME_VLA_TYPE}.yaml"
