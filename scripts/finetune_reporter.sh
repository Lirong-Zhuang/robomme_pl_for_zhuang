#!/usr/bin/env bash

# Fine-tune the Trinity Reporter. The selected JSONL uses the same prompt and
# two-image ordering as examples/robomme/reporter.py.
#
# Choose one of:
# /home/zhuanglr/robomme_pl_for_zhuang/data/trinity_preprocessed_data/reporter_data/reporter_qwenvl/simple_subgoal_train.jsonl
# /home/zhuanglr/robomme_pl_for_zhuang/data/trinity_preprocessed_data/reporter_data/reporter_qwenvl/grounded_subgoal_train.jsonl

REPORTER_DATASET_PATH='data/trinity_preprocessed_data/reporter_binfill_data_2/trainset/reporter_qwenvl/simple_subgoal_train.jsonl'
REPORTER_RUN_NAME='qwen_reporter_v4.1_simple_subgoal'
REPORTER_OUTPUT_DIR="/home/zhuanglr/robomme_pl_for_zhuang/runs/ckpts/reporter/${REPORTER_RUN_NAME}"

PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
IMAGE_MAX_TOKEN_NUM=256 \
VIDEO_MAX_TOKEN_NUM=64 \
FPS_MAX_FRAMES=10 \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model 'Qwen/Qwen3-VL-4B-Instruct' \
    --dataset "$REPORTER_DATASET_PATH" \
    --split_dataset_ratio 0.1 \
    --eval_strategy steps \
    --eval_steps 50 \
    --per_device_eval_batch_size 16 \
    --metric_for_best_model loss \
    --greater_is_better false \
    --load_best_model_at_end true \
    --load_from_cache_file true \
    --packing false \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 16 \
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
