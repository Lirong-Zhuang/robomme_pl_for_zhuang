# Personal dataset tools

Inspect one RoboMME HDF5 timestep:

```bash
uv run python user_code/inspect_h5_timestep.py \
  data/robomme_data_h5/record_dataset_BinFill.h5 \
  --episode 1 \
  --timestep 0 \
  --output-dir runs/h5_inspection
```

The command prints episode setup and timestep values, then creates:

```text
runs/h5_inspection/episode_0/timestep_0/
├── report.txt
├── summary.json
├── images/
│   └── *.png
└── arrays/
    └── *.npy
```

RGB arrays are saved directly as PNG. Depth and other image-shaped numerical
arrays are min-max normalized for PNG preview, while their lossless raw values
are also saved as NPY.

Export a complete episode:

```bash
uv run python user_code/inspect_h5_timestep.py \
  data/robomme_data_h5/record_dataset_BinFill.h5 \
  --episode 1 \
  --all-timesteps \
  --output-dir runs/h5_inspection
```

Inspect a preprocessed VLA PKL sample:

```bash
uv run python user_code/inspect_preprocessed_pkl.py \
  data/binfill_test_preprocessed \
  --sample 0 \
  --output-dir runs/pkl_inspection
```

Key state inspection 

```bash
uv run python user_code/export_h5_key_states_csv.py \
  data/robomme_data_h5/record_dataset_BinFill.h5 \
  --episode 1
```

Reporter test scoring

```bash
uv run python user_code/test_reporter.py
```

Set `CUDA_VISIBLE_DEVICES`, the Reporter checkpoint, the test-set path, and the
output directory in the `USER CONFIG` block at the top of
`user_code/test_reporter.py`. Command-line options remain available when a
one-off override is useful. The script evaluates one Reporter sequentially. The first
sample in each episode supplies the initial frame; later init frames are
updated only when the Reporter itself predicts `{"success": true}`. Exact
duplicates are disabled when `build_trainset_testset.py` creates the test set;
the evaluator also skips any duplicates found in older test sets as a safety
check. Completion is scored as causal subgoal progress: consecutive `true`
outputs are debounced, predictions up to two Reporter calls early are accepted,
delays up to two Reporter calls receive full credit,
delays of three or four calls are completed with a warning, and premature or
later transitions stop progress for that episode. During sequential inference,
both the init frame and active subgoal advance only on a debounced predicted
`true`; dataset labels never advance the prompt. The command writes
`summary.json`, `episode_completion.jsonl`, `predictions.jsonl`, `errors.json`,
and the corresponding `error_frames/` diagnostics.
