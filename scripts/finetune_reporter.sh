#!/usr/bin/env bash

# Fine-tune the Trinity Reporter. The selected JSONL uses the same prompt and
# two-image ordering as examples/robomme/reporter.py.
#
# Choose one of:
# /home/zhuanglr/robomme_pl_for_zhuang/data/trinity_preprocessed_data/reporter_data/reporter_qwenvl/simple_subgoal_train.jsonl
# /home/zhuanglr/robomme_pl_for_zhuang/data/trinity_preprocessed_data/reporter_data/reporter_qwenvl/grounded_subgoal_train.jsonl

REPORTER_DATASET_PATH='/home/zhuanglr/robomme_pl_for_zhuang/data/trinity_preprocessed_data/reporter_data/reporter_qwenvl/simple_subgoal_train.jsonl'
REPORTER_RUN_NAME='qwen_reporter_v1_simple_subgoal'
REPORTER_OUTPUT_DIR="/home/zhuanglr/robomme_pl_for_zhuang/runs/ckpts/reporter/${REPORTER_RUN_NAME}"

PYTORCH_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=256 \
VIDEO_MAX_TOKEN_NUM=64 \
FPS_MAX_FRAMES=10 \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model 'Qwen/Qwen3-VL-4B-Instruct' \
    --dataset "$REPORTER_DATASET_PATH" \
    --split_dataset_ratio 0.0 \
    --eval_strategy no \
    --load_from_cache_file true \
    --packing false \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 8 \
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
