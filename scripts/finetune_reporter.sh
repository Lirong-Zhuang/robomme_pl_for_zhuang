#!/usr/bin/env bash

set -Eeuo pipefail

# Fine-tune the Trinity Reporter. The selected JSONL uses the same prompt and
# image ordering as examples/robomme/reporter.py: one subgoal-init image,
# followed by 1..7 actual Reporter-call observations.

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

REPORTER_DATASET_PATH="data/trinity_preprocessed_data/reporter_binfill_data_5/trainset/reporter_qwenvl/simple_subgoal_train.jsonl"
REPORTER_RUN_NAME='qwen_reporter_v7_simple_subgoal'
REPORTER_OUTPUT_DIR="$REPO_ROOT/runs/ckpts/reporter/${REPORTER_RUN_NAME}"
RESUME_CHECKPOINT="$REPORTER_OUTPUT_DIR/v1-20261005-172357/checkpoint-100"

CUDA_DEVICE_IDS="0"
TRAIN_GLOBAL_BATCH_SIZE=16
# Evaluation does not update model weights. A smaller eval batch avoids a
# transient peak when the multimodal batch contains several long histories.
EVAL_GLOBAL_BATCH_SIZE=4
EVAL_AND_SAVE_STEPS=100

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

if [[ ! -f "$REPORTER_DATASET_PATH" ]]; then
    echo "ERROR: Reporter dataset not found: $REPORTER_DATASET_PATH" >&2
    exit 1
fi
if [[ ! -d "$RESUME_CHECKPOINT" ]]; then
    echo "ERROR: Resume checkpoint not found: $RESUME_CHECKPOINT" >&2
    exit 1
fi
if [[ ! -f "$RESUME_CHECKPOINT/trainer_state.json" ]]; then
    echo "ERROR: trainer_state.json not found in checkpoint: $RESUME_CHECKPOINT" >&2
    exit 1
fi

RUN_TIMESTAMP=$(date '+%Y%m%d-%H%M%S')
RUN_START_ISO=$(date --iso-8601=seconds)
LAUNCH_LOG_DIR="$REPORTER_OUTPUT_DIR/launcher_logs"
TRAIN_LOG="$LAUNCH_LOG_DIR/finetune-${RUN_TIMESTAMP}.log"
GPU_LOG="$LAUNCH_LOG_DIR/gpu-${RUN_TIMESTAMP}.csv"
FAILURE_LOG="$LAUNCH_LOG_DIR/failure-${RUN_TIMESTAMP}.log"
mkdir -p "$LAUNCH_LOG_DIR"

# Capture launcher messages and the complete Swift output while keeping Swift's
# actual exit status available to the failure-diagnostics block below.
exec > >(tee -a "$TRAIN_LOG") 2>&1

GPU_MONITOR_PID=""
monitor_gpu() {
    {
        echo "timestamp,index,uuid,name,memory.total MiB,memory.used MiB,memory.free MiB,utilization.gpu %,temperature.gpu C"
        while :; do
            nvidia-smi \
                --query-gpu=timestamp,index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
                --format=csv,noheader,nounits || true
            nvidia-smi \
                --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
                --format=csv,noheader,nounits || true
            sleep 5
        done
    } >>"$GPU_LOG" 2>&1
}

stop_gpu_monitor() {
    if [[ -n "$GPU_MONITOR_PID" ]] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
}
trap stop_gpu_monitor EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Reporter training configuration"
echo "  Dataset:               $REPORTER_DATASET_PATH"
echo "  Output:                $REPORTER_OUTPUT_DIR"
echo "  Resume checkpoint:     $RESUME_CHECKPOINT"
echo "  Visible GPUs:          $CUDA_DEVICE_IDS"
echo "  Train batch/GPU:       $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "  Eval batch/GPU:        $PER_DEVICE_EVAL_BATCH_SIZE"
echo "  Eval/save interval:    $EVAL_AND_SAVE_STEPS"
echo "  Complete training log: $TRAIN_LOG"
echo "  GPU monitor log:       $GPU_LOG"

if command -v nvidia-smi >/dev/null 2>&1; then
    monitor_gpu &
    GPU_MONITOR_PID=$!
else
    echo "WARNING: nvidia-smi is unavailable; GPU monitoring is disabled." | tee -a "$GPU_LOG"
fi

set +e
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
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
    --eval_steps "$EVAL_AND_SAVE_STEPS" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --eval_accumulation_steps 1 \
    --prediction_loss_only true \
    --metric_for_best_model loss \
    --greater_is_better false \
    --load_best_model_at_end true \
    --load_from_cache_file false \
    --packing false \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps 1 \
    --attn_impl sdpa \
    --padding_free false \
    --use_logits_to_keep true \
    --learning_rate 5e-5 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --save_strategy steps \
    --save_steps "$EVAL_AND_SAVE_STEPS" \
    --save_total_limit 2 \
    --logging_steps 50 \
    --max_length 3200 \
    --output_dir "$REPORTER_OUTPUT_DIR" \
    --run_name "$REPORTER_RUN_NAME" \
    --warmup_ratio 0.05 \
    --dataset_num_proc 8 \
    --dataloader_num_workers 4 \
    --resume_from_checkpoint "$RESUME_CHECKPOINT"
TRAIN_STATUS=$?
set -e

stop_gpu_monitor

if (( TRAIN_STATUS != 0 )); then
    {
        echo "Reporter training failed with exit status $TRAIN_STATUS at $(date --iso-8601=seconds)."
        echo
        echo "Potential root-cause lines from the training log:"
        grep -nEi -B 20 -A 60 \
            'outofmemory|out of memory|traceback|runtimeerror|acceleratorerror|exception|killed|no space left|nccl|xid|childfailederror' \
            "$TRAIN_LOG" | tail -n 1200 || true
        echo
        echo "Last 300 lines of the training log:"
        tail -n 300 "$TRAIN_LOG" || true
        echo
        echo "Current NVIDIA state:"
        nvidia-smi -q 2>&1 || true
        echo
        echo "Current host memory:"
        free -h 2>&1 || true
        echo
        echo "Current filesystem usage:"
        df -h "$REPO_ROOT" 2>&1 || true
        echo
        echo "Kernel messages recorded since launch:"
        journalctl -k --since "$RUN_START_ISO" --no-pager 2>&1 | tail -n 500 || true
    } >"$FAILURE_LOG"
    echo "ERROR: Reporter training failed with exit status $TRAIN_STATUS." >&2
    echo "Failure diagnosis: $FAILURE_LOG" >&2
    echo "Complete training log: $TRAIN_LOG" >&2
    echo "GPU monitor log: $GPU_LOG" >&2
    exit "$TRAIN_STATUS"
fi

echo "Reporter training completed successfully."
echo "Complete training log: $TRAIN_LOG"
echo "GPU monitor log: $GPU_LOG"
