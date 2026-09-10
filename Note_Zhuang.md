# Trinity

This branch develops a benchmark with three separated parts to do the memory-based tasks.

## 1. Build the dataset

The QwenVL builder writes the processed dataset under
`<preprocessed_data_path>/qwenvl/`. The training JSONL files are:

```text
qwenvl/simple_subgoal_train.jsonl
qwenvl/grounded_subgoal_train.jsonl
```

### `data/` and `runs/` layout

All paths below are relative to the repository root. Names in angle brackets
are supplied by the corresponding build, training, or evaluation command.

```text
data/
├── robomme_data_h5/                         # optional local raw RoboMME HDF5 data
│   └── <TaskName>.h5
├── robomme_preprocessed_data/               # Executer training dataset
│   ├── data/
│   │   └── <sample_id>.pkl
│   ├── features/
│   │   └── episode_<episode_id>/
│   │       ├── token_emb_<step>.npy
│   │       └── kept_indices.json
│   └── meta/
│       └── stats.json
└── trinity_preprocessed_data/               # Manager training datasets
    └── <manager_dataset_name>/
        └── qwenvl/
            ├── images/
            │   ├── <TaskName>_ep<episode>_step<step>.png
            │   └── <TaskName>_ep<episode>_video.mp4
            ├── simple_subgoal_train.jsonl
            └── grounded_subgoal_train.jsonl

runs/
├── assets/
│   └── mme_vla_suite/                       # Executer normalization/assets data
├── ckpts/
│   ├── manager/
│   │   └── <manager_run_name>/
│   │       └── <version>/
│   │           └── checkpoint-<step>/       # Manager LoRA checkpoint
│   ├── executer/
│   │   └── <executer_run_name>/
│   │       └── <step>/                      # newly trained Trinity Executer
│   └── mme_vla_suite/
│       └── symbolic-simple-subgoal/
│           └── 79999/                       # legacy rep Executer checkpoint
└── evaluation/
    ├── server_logs/
    └── <executer_name>/
        └── <evaluation_run_name>/             # e.g. trinity_v1.1
            ├── summary.json                   # aggregate of all seeds/repeats
            └── seed<seed>/
                └── repeat<repeat_id>/
                    ├── progress.json
                    ├── log.json
                    └── <TaskName>/
                        ├── frames/
                        │   └── ep<episode_id>/
                        │       └── step_<step>_image.png
                        ├── init_frames/
                        │   └── ep<episode_id>/
                        │       └── step_<step>_image.png
                        ├── manager_logs/
                        │   └── <TaskName>_ep<episode_id>.log
                        ├── reporter_logs/
                        │   └── <TaskName>_ep<episode_id>.log
                        └── videos/
                            └── <evaluation_video>.mp4
```

### Build all `.h5` files

Omit `--tasks` to process every `.h5` file directly under
`--raw_data_path`:

```bash
uv run python scripts/build_dataset.py \
  --dataset_type manager_qwenvl \
  --raw_data_path /data/public/RoboMME \
  --preprocessed_data_path data/trinity_preprocessed_data/manager_all_data
```

### Build only specified tasks

Use `--tasks` followed by one or more task names. This example processes only
the `BinFill` dataset:

```bash
uv run python scripts/build_dataset.py \
  --dataset_type manager_qwenvl \
  --raw_data_path /data/public/RoboMME \
  --preprocessed_data_path data/trinity_preprocessed_data/manager_binfill_data_1 \
  --tasks BinFill
```

For multiple tasks, list their names separated by spaces, for example:

```bash
--tasks BinFill PickXtimes
```

Add `--visualize` only when visualization MP4s are needed. Keyframe duplicate
training samples are generated whether or not visualization is enabled.

### Build Reporter data

Reporter data uses the same selected frames and transition-frame duplication
as the QwenVL Manager builder. Within each subgoal span, the first observation
is the subgoal start frame. Selected intermediate observations are labelled
`{"success": false}` and the subgoal transition observation is labelled
`{"success": true}`. The terminal episode frame is excluded because the live
Reporter is not called after the environment terminates. Each subgoal owns a
capacity-`k` sliding window containing only Reporter-call observations, not
adjacent raw timesteps. The default is `k=7`. The window starts empty, is not
padded while filling, and is cleared immediately when the subgoal completes.

```bash
uv run python scripts/build_dataset.py \
  --dataset_type reporter_qwenvl \
  --raw_data_path /data/public/RoboMME \
  --preprocessed_data_path /home/zhuanglr/robomme_pl_for_zhuang/data/trinity_preprocessed_data/reporter_data \
  --reporter_history_size 7
```

The builder writes:

```text
reporter_data/reporter_qwenvl/images/
reporter_data/reporter_qwenvl/simple_subgoal_train.jsonl
reporter_data/reporter_qwenvl/grounded_subgoal_train.jsonl
```

Each JSONL row contains `system`, `user`, and `assistant` messages plus 2–8
images by default: one current-subgoal init observation followed by the 1–7
Reporter-call observations currently available, ordered oldest to newest with
the current observation last. The prompt is imported from the same shared
module as live Reporter inference, preventing training/inference prompt drift.

Before training, select the simple or grounded JSONL and GPU settings near the
top of `scripts/finetune_reporter.sh`. `REPORTER_HISTORY_SIZE` must match data
generation; the script validates the first row before starting training.
Then run:

```bash
bash scripts/finetune_reporter.sh
```

To evaluate a trained LoRA, set `REPORTER_ADAPTER_PATH` in `scripts/eval.sh`
to the generated checkpoint directory and keep `REPORTER_HISTORY_SIZE` equal
to the training value. Old two-image Reporter adapters should not be reused.

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

- `EVAL_PRESET`
- `EXECUTER_CKPT_ID`
- `EXECUTER_SEEDS` and `NUM_REPEATS`
- `ONLY_TASKS` and `NUM_EPISODES`
- the Manager and Reporter adapter paths
- `EXECUTER_GPU_ID` and `MANAGER_REPORTER_GPU_ID`
- `FRAMEWORK_VERSION`, `RUN_NAME`, and `SAVE_DIR`

Run the evaluation from the repository root:

```bash
bash scripts/eval.sh
```

The script evaluates every configured seed/repeat pair sequentially. It starts
a fresh policy server for each pair, writes the individual result under
`<SAVE_DIR>/<executer_name>/<evaluation_run_name>/seed<seed>/repeat<repeat>/`,
updates `summary.json` in the evaluation-run directory, and stops the server
automatically. Re-running with `OVERWRITE=false` resumes incomplete episodes.
