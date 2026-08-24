# Fine-tune the Trinity Manager.
# More usage of Swift, please refer to https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html


# Choose the dataset path from the following list, and change the OUTPUT_DIR accordingly:
# data/robomme_preprocessed_data/qwenvl/simple_subgoal_train.jsonl
# data/robomme_preprocessed_data/qwenvl/grounded_subgoal_train.jsonl
# data/robomme_preprocessed_data/memer/grounded_subgoal_train.jsonl

MANAGER_DATASET_PATH='data/trinity_preprocessed_data/manager_binfill_data_1/qwenvl/simple_subgoal_train.jsonl'
MANAGER_RUN_NAME='qwen_manager_v2_simple_subgoal'
MANAGER_OUTPUT_DIR="runs/ckpts/manager/${MANAGER_RUN_NAME}"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=256 \
VIDEO_MAX_TOKEN_NUM=64 \
FPS_MAX_FRAMES=10 \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model 'Qwen/Qwen3-VL-4B-Instruct' \
    --dataset $MANAGER_DATASET_PATH \
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
    --output_dir $MANAGER_OUTPUT_DIR \
    --run_name $MANAGER_RUN_NAME \
    --warmup_ratio 0.05 \
    --dataset_num_proc 8 \
    --dataloader_num_workers 4
