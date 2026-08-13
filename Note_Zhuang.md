# Trinity

This branch develops a benchmark with three separated parts to do the memory-based tasks.

## 1. Build the dataset

The QwenVL builder writes the processed dataset under
`<preprocessed_data_path>/qwenvl/`. The training JSONL files are:

```text
qwenvl/simple_subgoal_train.jsonl
qwenvl/grounded_subgoal_train.jsonl
```

### Build all `.h5` files

Omit `--tasks` to process every `.h5` file directly under
`--raw_data_path`:

```bash
uv run python scripts/build_dataset.py \
  --dataset_type vlm_subgoal_qwenvl \
  --raw_data_path /data/public/RoboMME \
  --preprocessed_data_path data/robomme_preprocessed_data
```

### Build only specified tasks

Use `--tasks` followed by one or more task names. This example processes only
the `BinFill` dataset:

```bash
uv run python scripts/build_dataset.py \
  --dataset_type vlm_subgoal_qwenvl \
  --raw_data_path /data/public/RoboMME \
  --preprocessed_data_path data/robomme_preprocessed_binfill_data \
  --tasks BinFill
```

For multiple tasks, list their names separated by spaces, for example:

```bash
--tasks BinFill PickXtimes
```

Add `--visualize` only when visualization MP4s are needed. Keyframe duplicate
training samples are generated whether or not visualization is enabled.

## 2. Train the VLM subgoal predictor with tmux

Before training, set `DATASET_PATH`, `RUN_NAME`, `OUTPUT_DIR`, and, if needed,
`CUDA_VISIBLE_DEVICES` in `scripts/finetune_vlm_subgoal_predictor.sh`.

Create and enter a named tmux session:

```bash
tmux new -s vlm_train
```

Initialize micromamba and activate the `robomme` environment inside that
session:

```bash
micromamba activate robomme
```

Then start training:

```bash
bash scripts/finetune_vlm_subgoal_predictor.sh
```

Detach without stopping training by pressing:

```text
Ctrl-b, then d
```

List running tmux sessions and check whether the training session still
exists:

```bash
tmux ls
```

Enter the training session again:

```bash
tmux attach -t vlm_train
```

If the session is already attached elsewhere, detach it there and attach it
to the current terminal:

```bash
tmux attach -d -t vlm_train
```

After training has finished, the session can be closed from inside it with
`exit`, or from another terminal with:

```bash
tmux kill-session -t vlm_train
```

## 3. Evaluation

Before evaluation, edit the **Evaluation configuration** section near the top
of `scripts/eval.sh`. In particular, check:

- `MODEL_TYPE`
- `CKPT_ID`
- `ONLY_TASKS` and `NUM_EPISODES`
- `QWENVL_SIMPLE_ADAPTER_PATH` or the adapter path for the selected predictor
- `GPU_ID_SERVER` and `GPU_ID_CLIENT`
- `EVAL_RUN_NAME` and `SAVE_DIR`

Run the evaluation from the repository root:

```bash
bash scripts/eval.sh
```

The script starts the policy server, runs the evaluation in the foreground,
writes results under `SAVE_DIR`, and stops the server automatically when the
evaluation ends.
