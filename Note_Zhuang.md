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

The original Hugging Face HDF5 files remain unchanged. All new simulator
rollouts are stored in one raw HDF5 pool. Train/test membership is assigned
later, when Reporter JSONL/index files are built, using an episode-level split.

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
7. Keep raw HDF5 together; create an episode-disjoint failure-recovery test
   split only in the derived Reporter JSONL/index.

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

Use the following layout on the training machine. The raw HDF5 pool itself has
no train/test subdirectories:

```text
data/
├── robomme_data_h5/                  # downloaded Hugging Face HDF5
├── h5_data/                          # failure-recovery raw pool
│   ├── hdf5_files/
│   ├── videos/
│   └── merged/                       # lossless packaging, still no split
│       └── record_dataset_BinFill_with_failure.h5
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

## Failure-recovery HDF5 generation

Run the generator with the RoboMME simulator environment rather than the root
policy-training environment. `uv --project` selects the submodule environment.
First edit `SERVER_OUTPUT_DIR` near the top of
`scripts/generate_failure_recovery_h5.py`, for example:

```python
SERVER_OUTPUT_DIR = "/data/zhuang/binfill_failure_recovery"
```

First generate episode 0 as a smoke test and inspect its video:

```bash
uv run --project third_party/robomme_benchmark \
  python scripts/generate_failure_recovery_h5.py \
  --task BinFill --failure-mode z --difficulty easy \
  --num-episodes 1 --start-episode 0 --base-seed 100000 --save-video
```

The complete 30-episode batch commands, including the recommended mix of `z`,
`xy`, and three difficulties, are documented at the very top of the generator.
Every command uses a distinct episode range so all outputs can coexist in one
directory.

The path may still be overridden for a one-off run without editing the file:

```bash
uv run --project third_party/robomme_benchmark \
  python scripts/generate_failure_recovery_h5.py \
  --task BinFill --failure-mode z --num-episodes 1 \
  --output-dir /absolute/temporary/server/path --save-video
```

The generator retries different simulator seeds until each logical episode
finishes successfully. All modes and difficulties share one output pool:

```text
SERVER_OUTPUT_DIR/
├── generation_manifest_z_easy_ep0_to_0.json
├── generation_manifest_xy_hard_ep26_to_29.json
├── hdf5_files/BinFill_ep0_seed100000.h5
├── hdf5_files/BinFill_ep29_seed....h5
├── videos/...FailRecoverZ....mp4
└── videos/...FailRecoverXY....mp4
```

The generator removes empty HDF5 containers left by rejected attempts. It will
not replace an existing task/episode/seed file unless `--overwrite` is passed.

## Where the generation method comes from

This command is a branch-level production wrapper assembled from code in the
official `RoboMME/robomme_benchmark` submodule; it is not a command copied from
the public README:

- `src/robomme/robomme_env/utils/task4recovery.py` implements failed-grasp task
  injection.
- `src/robomme/robomme_env/utils/planner-ref.py` implements the `z` and `xy`
  failed pickup followed by the normal pickup.
- `src/robomme/env_record_wrapper/RecordWrapper.py` writes successful episodes
  to per-episode HDF5 files and optional videos.
- `tests/_shared/dataset_generation.py` shows how RoboMME's own tests create an
  environment, enable recovery, run the oracle, retry seeds, and record data.

The public README does not document a usable generation command. Its Data
Generation section is inside an HTML comment and still contains the placeholder
`uv run scripts/dev/xxxx`. The test README describes the existing helper as a
pytest fixture/cache that writes temporary test data. Therefore this branch
adds `scripts/generate_failure_recovery_h5.py` as a stable server entry point
and adds safety checks and manifests around the upstream primitives.

## Merge the accepted raw episodes

After validating all 30 raw episodes, combine them with the downloaded BinFill
HDF5. The server copy currently contains 100 normal episodes, so this packaging
step creates one 130-episode source pool and does not assign train/test
membership. Original episode IDs 0-99 are retained; recovery episode IDs 0-29
are remapped to 100-129 to prevent collisions. The generation mode, difficulty,
seed, and attempt count are copied from the manifests into each recovery
`episode_*` group's attributes:

```bash
uv run --project third_party/robomme_benchmark \
  python scripts/merge_episode_h5.py \
  --base-h5 /data/public/RoboMME/record_dataset_BinFill.h5 \
  --base-expected-count 100 \
  --input-dir data/h5_data/hdf5_files \
  --manifest-dir data/h5_data \
  --task BinFill \
  --expected-count 30 \
  --expected-start-episode 0 \
  --output data/h5_data/merged/record_dataset_BinFill_with_failure.h5
```

The merger refuses incomplete episode ranges, duplicate episode IDs,
manifest/HDF5 seed mismatches, unexpected base counts, and accidental
overwrite. Both the downloaded HDF5 and raw per-episode files remain unchanged.

## Planned Reporter conversion

The Reporter JSONL command below describes the following stage of the branch.
It must only be used after its dataset builder/options have been implemented.

Build an episode-disjoint Reporter train/test split:

```bash
uv run python scripts/build_trainset_testset.py \
  --dataset_type reporter_qwenvl \
  --raw_data_path data/h5_data/merged \
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

- Keep raw HDF5 together; split by complete episode before sampling or
  augmentation when constructing Reporter JSONL/index files.
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
