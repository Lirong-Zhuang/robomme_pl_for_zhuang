# `rep_dataset_generation` branch notes

## Purpose

This branch is for building Reporter training and evaluation data that includes
intentional grasp failures. The immediate problem is that the Reporter can
mistake a completed-looking robot motion for a completed object-centric
subgoal. In particular, an empty grasp may close and lift the gripper without
lifting the target object.

The intended data pipeline is:

```text
original Hugging Face HDF5 (normal successful trajectories)
                              \
                               +--> Reporter JSONL --> train Reporter
                              /
new failure-recovery HDF5 (failed grasp followed by successful retry)
```

The original Hugging Face HDF5 files remain unchanged. New simulator rollouts
are stored separately and mixed with the original data only after episode-level
train/test splitting.

## Branch scope

The planned work is:

1. Add a production data-generation entry point based on RoboMME's existing
   `robomme_failure_recovery` environment option.
2. Generate both `z` and `xy` failed-grasp variants with different seeds.
3. Record only episodes that eventually recover and finish successfully.
4. Merge the per-episode HDF5 files into one HDF5 file per task.
5. Build Reporter JSONL samples with the exact same image-window format used at
   inference time.
6. Guarantee that the completed-looking failed grasp is included as a hard
   negative (`{"success": false}`).
7. Keep a failure-recovery test split that is never mixed into training.

This branch should initially avoid unrelated changes to the Manager, Executer,
and policy evaluation logic.

## RoboMME behavior being reused

The upstream benchmark submodule already contains failure injection:

- `robomme_failure_recovery=True` enables it.
- `robomme_failure_recovery_mode="z"` stops above the true grasp pose.
- `robomme_failure_recovery_mode="xy"` offsets the grasp laterally.
- The injected solver performs an empty grasp and then continues with the
  normal grasp, allowing the overall episode to finish successfully and be
  written to HDF5.

The existing implementation can be found in:

```text
third_party/robomme_benchmark/src/robomme/robomme_env/utils/task4recovery.py
third_party/robomme_benchmark/src/robomme/robomme_env/utils/planner-ref.py
third_party/robomme_benchmark/tests/_shared/dataset_generation.py
```

The pytest helper is a reference implementation, not the intended production
dataset command. It enables recovery only for a small fixed episode range and
writes through pytest's temporary cache.

## Local directory layout

Use the following layout on the training machine:

```text
data/
├── robomme_data_h5/                  # downloaded Hugging Face HDF5
├── failure_recovery_raw/             # per-episode simulator outputs
│   ├── hdf5_files/
│   └── videos/
├── failure_recovery_merged/          # one merged HDF5 per task
│   └── data_BinFill.h5
└── reporter_failure_recovery/        # generated train/test JSONL and images
```

All of these paths are ignored by Git. Source code, tests, and this note remain
tracked.

## Existing setup and verification commands

Install the root project and initialize the benchmark submodule:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
git submodule update --init
```

Verify the lightweight benchmark logic:

```bash
cd third_party/robomme_benchmark
uv run python -m pytest tests/lightweight/
cd ../..
```

Run the upstream physics-and-recording dataset tests when the simulator is
available. These tests may be slow and use a temporary dataset cache:

```bash
cd third_party/robomme_benchmark
uv run python -m pytest tests/dataset/ -s
cd ../..
```

Build the original repository's existing preprocessing outputs from downloaded
HDF5 data:

```bash
uv run scripts/build_dataset.py \
  --dataset_type robomme_pkl \
  --raw_data_path data/robomme_data_h5 \
  --preprocessed_data_path data/robomme_preprocessed_data
```

The current `main`-branch builder supports `robomme_pkl`,
`vlm_subgoal_qwenvl`, and `vlm_subgoal_memer`. It does not yet support Reporter
JSONL.

## Planned production commands

The following commands describe the target workflow for this branch. They must
only be used after the corresponding scripts/options have been implemented.

Generate failed-grasp/recovery episodes for both failure modes:

```bash
uv run python scripts/generate_failure_recovery_h5.py \
  --task BinFill \
  --failure-mode z \
  --num-episodes 50 \
  --output-dir data/failure_recovery_raw \
  --save-video

uv run python scripts/generate_failure_recovery_h5.py \
  --task BinFill \
  --failure-mode xy \
  --num-episodes 50 \
  --output-dir data/failure_recovery_raw \
  --save-video
```

Merge successful per-episode files into the filename/layout expected by the
dataset builders:

```bash
uv run python scripts/merge_episode_h5.py \
  --input-dir data/failure_recovery_raw/hdf5_files \
  --task BinFill \
  --output data/failure_recovery_merged/data_BinFill.h5
```

Build an episode-disjoint Reporter train/test split:

```bash
uv run python scripts/build_trainset_testset.py \
  --dataset_type reporter_qwenvl \
  --raw_data_path data/failure_recovery_merged \
  --preprocessed_data_path data/reporter_failure_recovery \
  --tasks BinFill \
  --reporter_history_size 7 \
  --test_ratio 0.2 \
  --seed 42 \
  --visualize
```

Before training, inspect the visualization and confirm that every selected hard
negative shows all of the following:

1. the gripper attempted and visually finished a grasp motion;
2. the target object was not lifted or held;
3. the active subgoal remained the same;
4. the label is `{"success": false}`;
5. the later genuine grasp is labelled `{"success": true}`.

## Data-splitting rules

- Split by complete episode before sampling or augmentation.
- Never train on the held-out failure-recovery episodes.
- Do not infer a per-subgoal label from the final episode outcome alone.
- Do not use samples after a false-positive Reporter transition unless they are
  relabelled from simulator ground truth.
- Avoid exact duplication of positive transition rows.
- Start with roughly 25-40% failure-recovery hard negatives in the Reporter
  training mixture, then tune using held-out false-positive rate.

## Acceptance criteria

The branch is ready for Reporter training when:

- generation is deterministic for a fixed seed;
- both `z` and `xy` modes are represented;
- generated videos visibly contain the intended empty grasp;
- failed-grasp endpoint frames are selected explicitly rather than only by a
  fixed stride;
- generated JSONL image order matches live Reporter inference;
- train and test episodes are disjoint;
- a held-out failed-grasp false-positive rate can be reported separately from
  ordinary completion accuracy.
