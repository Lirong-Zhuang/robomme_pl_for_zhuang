#!/usr/bin/env bash

# Fine-tune the Trinity Reporter. The selected JSONL uses the same prompt and
# image ordering as examples/robomme/reporter.py: one subgoal-init image,
# followed by 1..7 actual Reporter-call observations.
#
# Choose one of:
# /home/zhuanglr/robomme_pl_for_zhuang/data/trinity_preprocessed_data/reporter_data/reporter_qwenvl/simple_subgoal_train.jsonl
# /home/zhuanglr/robomme_pl_for_zhuang/data/trinity_preprocessed_data/reporter_data/reporter_qwenvl/grounded_subgoal_train.jsonl

REPORTER_DATASET_PATH='data/trinity_preprocessed_data/reporter_binfill_data_2/trainset/reporter_qwenvl/simple_subgoal_train.jsonl'
REPORTER_RUN_NAME='qwen_reporter_v5_simple_subgoal'
REPORTER_OUTPUT_DIR="/home/zhuanglr/robomme_pl_for_zhuang/runs/ckpts/reporter/${REPORTER_RUN_NAME}"
CUDA_DEVICE_IDS="0"
TRAIN_GLOBAL_BATCH_SIZE=16
EVAL_GLOBAL_BATCH_SIZE=32

IFS=',' read -r -a CUDA_DEVICE_ID_LIST <<< "$CUDA_DEVICE_IDS"
NUM_GPUS=${#CUDA_DEVICE_ID_LIST[@]}
if ((
    TRAIN_GLOBAL_BATCH_SIZE % NUM_GPUS != 0
    || EVAL_GLOBAL_BATCH_SIZE % NUM_GPUS != 0
)); then
    echo "ERROR: train/eval global batch sizes must be divisible by NUM_GPUS=${NUM_GPUS}." >&2
    exit 1
fi
PER_DEVICE_TRAIN_BATCH_SIZE=$((TRAIN_GLOBAL_BATCH_SIZE / NUM_GPUS))
PER_DEVICE_EVAL_BATCH_SIZE=$((EVAL_GLOBAL_BATCH_SIZE / NUM_GPUS))

PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
IMAGE_MAX_TOKEN_NUM=256 \
VIDEO_MAX_TOKEN_NUM=64 \
FPS_MAX_FRAMES=10 \
NPROC_PER_NODE="$NUM_GPUS" \
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_IDS" \
swift sft \
    --model 'Qwen/Qwen3-VL-4B-Instruct' \
    --dataset "$REPORTER_DATASET_PATH" \
    --split_dataset_ratio 0.1 \
    --eval_strategy steps \
    --eval_steps 50 \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --metric_for_best_model loss \
    --greater_is_better false \
    --load_best_model_at_end true \
    --load_from_cache_file true \
    --packing false \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 4 \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps 1 \
    --attn_impl sdpa \
    --padding_free false \
    --learning_rate 5e-5 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 50 \
    --max_length 3200 \
    --output_dir "$REPORTER_OUTPUT_DIR" \
    --run_name "$REPORTER_RUN_NAME" \
    --warmup_ratio 0.05 \
    --dataset_num_proc 8 \
    --dataloader_num_workers 4
