# A total of 14 VLA variants are considered in our experiments:
#  FrameSamp                        TokenDrop                       RMT                       TTT                      Symbolic
# perceptual-framesamp-context  perceptual-tokendrop-context  recurrent-rmt-context  recurrent-ttt-context  symbolic-grounded-subgoal
# perceptual-framesamp-expert   perceptual-tokendrop-expert   recurrent-rmt-expert   recurrent-ttt-expert   symbolic-simple-subgoal
# perceptual-framesamp-modul    perceptual-tokendrop-modul    recurrent-rmt-modul    recurrent-ttt-modul

EXECUTER_TYPE="symbolic-simple-subgoal"

# Keep the internal training config ID configurable. Its current value remains
# "mme_vla_suite" for checkpoint and norm-stat compatibility.
EXECUTER_TRAIN_CONFIG="mme_vla_suite"
EXECUTER_RUN_NAME="${EXECUTER_TYPE}_your_model_name"
EXECUTER_CHECKPOINT_NAMESPACE="executer"

export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
uv run scripts/train.py "$EXECUTER_TRAIN_CONFIG" \
--exp-name="$EXECUTER_RUN_NAME" \
--checkpoint-namespace="$EXECUTER_CHECKPOINT_NAMESPACE" \
--batch-size=64 \
--num-workers=4 \
--fsdp-devices=4 \
--dataset-path=data/robomme_preprocessed_data \
--model.use_history \
--model.history_config="${EXECUTER_TYPE}.yaml"
