# Fine-tune the Executer with LoRA on the 2B VLM expert while keeping the
# 300M action expert and the action projections fully trainable.

# A total of 14 VLA variants are considered in our experiments:
#  FrameSamp                        TokenDrop                       RMT                       TTT                      Symbolic
# perceptual-framesamp-context  perceptual-tokendrop-context  recurrent-rmt-context  recurrent-ttt-context  symbolic-grounded-subgoal
# perceptual-framesamp-expert   perceptual-tokendrop-expert   recurrent-rmt-expert   recurrent-ttt-expert   symbolic-simple-subgoal
# perceptual-framesamp-modul    perceptual-tokendrop-modul    recurrent-rmt-modul    recurrent-ttt-modul

EXECUTER_TYPE="symbolic-simple-subgoal"

# Experiment and data settings.
EXECUTER_TRAIN_CONFIG="mme_vla_suite_lora"
EXECUTER_RUN_NAME="excuter_pi0.5_v2"
EXECUTER_CHECKPOINT_NAMESPACE="executer"
EXECUTER_DATASET_PATH="data/trinity_preprocessed_data/executer_binfill_data_0"

# Hardware settings. Keep FSDP_DEVICES equal to the number of visible GPUs.
# Examples: GPU_IDS="0" and FSDP_DEVICES=1; GPU_IDS="0,1" and FSDP_DEVICES=2.
GPU_IDS="0"
FSDP_DEVICES=1

# Training settings.
BATCH_SIZE=64
NUM_TRAIN_STEPS=80000
NUM_WORKERS=4
SEED=42
LOG_INTERVAL=100
SAVE_INTERVAL=10000
KEEP_PERIOD=10000

# export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"

CUDA_VISIBLE_DEVICES="$GPU_IDS" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
uv run scripts/train.py "$EXECUTER_TRAIN_CONFIG" \
    --exp-name="$EXECUTER_RUN_NAME" \
    --checkpoint-namespace="$EXECUTER_CHECKPOINT_NAMESPACE" \
    --batch-size="$BATCH_SIZE" \
    --num-train-steps="$NUM_TRAIN_STEPS" \
    --num-workers="$NUM_WORKERS" \
    --fsdp-devices="$FSDP_DEVICES" \
    --seed="$SEED" \
    --log-interval="$LOG_INTERVAL" \
    --save-interval="$SAVE_INTERVAL" \
    --keep-period="$KEEP_PERIOD" \
    --dataset-path="$EXECUTER_DATASET_PATH" \
    --model.use-history \
    --resume \
    --model.history-config="${EXECUTER_TYPE}.yaml"
