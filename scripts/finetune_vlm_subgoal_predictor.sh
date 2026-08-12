# More usage of Swift, please refer to https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html


# Choose the dataset path from the following list, and change the OUTPUT_DIR accordingly:
# data/robomme_preprocessed_data/qwenvl/simple_subgoal_train.jsonl
# data/robomme_preprocessed_data/qwenvl/grounded_subgoal_train.jsonl
# data/robomme_preprocessed_data/memer/grounded_subgoal_train.jsonl

DATASET_PATH='/data/zhuanglr/robomme_preprocessed_data/qwenvl/grounded_subgoal_train.jsonl'
RUN_NAME='qwenvl_grounded_subgoal_256_v1.2'
OUTPUT_DIR="runs/ckpts/vlm_subgoal_predictor/qwenvl/${RUN_NAME#qwenvl_}"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=256 \
VIDEO_MAX_TOKEN_NUM=64 \
FPS_MAX_FRAMES=10 \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model 'Qwen/Qwen3-VL-4B-Instruct' \
    --dataset $DATASET_PATH \
    --norm_bbox none \
    --split_dataset_ratio 0.1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --per_device_eval_batch_size 16 \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --load_best_model_at_end true \
    --load_from_cache_file false \
    --packing false \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --attn_impl sdpa \
    --padding_free false \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_checkpointing false \
    --vit_gradient_checkpointing false \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 100 \
    --max_length 3200 \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME \
    --warmup_ratio 0.05 \
    --dataset_num_proc 8 \
    --dataloader_num_workers 8
