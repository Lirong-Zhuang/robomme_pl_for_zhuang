# A total of 14 VLA variants are considered in our experiments:
#  FrameSamp                        TokenDrop                       RMT                       TTT                      Symbolic
# perceptual-framesamp-context  perceptual-tokendrop-context  recurrent-rmt-context  recurrent-ttt-context  symbolic-grounded-subgoal
# perceptual-framesamp-expert   perceptual-tokendrop-expert   recurrent-rmt-expert   recurrent-ttt-expert   symbolic-simple-subgoal
# perceptual-framesamp-modul    perceptual-tokendrop-modul    recurrent-rmt-modul    recurrent-ttt-modul

MME_VLA_TYPE="symbolic-simple-subgoal"

# export WANDB_API_KEY=<YOUR_WANDB_API_KEY>

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
uv run scripts/train.py mme_vla_suite \
--exp-name=${MME_VLA_TYPE}_your_model_name \
--batch-size=64 \
--num-workers=4 \
--fsdp-devices=4 \
--dataset-path=data/robomme_preprocessed_dup_binfill_data \
--model.use_history \
--model.history_config="${MME_VLA_TYPE}.yaml"
