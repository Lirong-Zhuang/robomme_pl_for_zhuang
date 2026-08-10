
## Automat

change line 25-84 of file scripts/eval.sh to change the setup

```bash
bash scripts/eval.sh
```

## tmux

Create a named tmux session:

```bash
tmux new -s vlm_train
```

Detach from the current session without stopping the training process:

```text
Ctrl-b, then d
```

List existing tmux sessions:

```bash
tmux ls
```

Reattach to the training session:

```bash
tmux attach -t vlm_train
```

Terminate the session after training has finished:

```bash
tmux kill-session -t vlm_train
```

## Train the VLM subgoal predictor

Before training, edit `DATASET_PATH` and `OUTPUT_DIR` in
`scripts/finetune_vlm_subgoal_predictor.sh`. The script is configured to use
one GPU (`CUDA_VISIBLE_DEVICES=0`).

Start training inside tmux:

```bash
tmux new -s vlm_train
bash scripts/finetune_vlm_subgoal_predictor.sh
```

To use a different GPU, change `CUDA_VISIBLE_DEVICES=0` in the finetune
script to the required device index before starting the run.

The configured training dataset and checkpoints are written according to:

```bash
DATASET_PATH='data/robomme_preprocessed_data/qwenvl/simple_subgoal_train.jsonl'
OUTPUT_DIR='runs/ckpts/vlm_subgoal_predictor/qwenvl/simple_subgoal'
```

## Manuel

This project uses two separate Python environments.

### 1. Policy Learning / uv Environment

Use this environment for the VLA policy learning code and policy server.

Activate:

```bash
source .venv/bin/activate
```

Deactivate:

```bash
deactivate
```

Typical usage:

```bash
uv run scripts/serve_policy.py ...
```

### 2. RoboMME Simulator / conda Environment

Use this environment for the RoboMME simulator, `simple_test.py`, and `eval.py`.

Activate:

```bash
conda activate robomme
```

Deactivate:

```bash
conda deactivate
```

Typical usage:

```bash
python examples/robomme/simple_test.py
python examples/robomme/eval.py ...
```

## Run a eval to observe the response

change line 47-64 of file examples/robimme/eval.py to change the setup

### Terminal 1 with uv env(openpi)

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py
```

### Terminal 2 with conda env(robomme)

```bash
CUDA_VISIBLE_DEVICES=0 python examples/robomme/eval.py
```
