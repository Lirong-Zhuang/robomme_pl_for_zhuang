"""Build episode-disjoint train and test datasets from RoboMME HDF5 files.

This entry point uses the existing Executer, Manager, and Reporter builders
unchanged after assigning complete episodes to one split.  By default, each
selected task contributes 10% of its episodes to ``testset`` and all remaining
episodes to ``trainset``.

Example:

```
uv run python scripts/build_trainset_testset.py \
  --dataset_type reporter_qwenvl \
  --raw_data_path /data/public/RoboMME \
  --preprocessed_data_path data/trinity_preprocessed_data/reporter_binfill_data_2 \
  --tasks BinFill \
  --test_ratio 0.1 \
  --seed 42
```
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import h5py

from mme_vla_suite.dataset_builder.build_executer_dataset import (
    ExecuterDatasetProcessor,
)
from mme_vla_suite.dataset_builder.build_manager_dataset_memer import (
    DatasetBuilder as MemerManagerDatasetBuilder,
)
from mme_vla_suite.dataset_builder.build_manager_dataset_qpa import (
    QPA_DIR_NAME,
    DatasetBuilder as QPAManagerDatasetBuilder,
)
from mme_vla_suite.dataset_builder.build_manager_dataset_qwenvl import (
    DatasetBuilder as QwenVLManagerDatasetBuilder,
)
from mme_vla_suite.dataset_builder.build_reporter_dataset import (
    DatasetBuilder as ReporterDatasetBuilder,
)
from mme_vla_suite.dataset_builder.robomme_h5_utils import (
    get_env_id_from_filename,
    get_episode_indices,
)


DATASET_TYPES = (
    "executer",
    "manager_qwenvl",
    "manager_qpa",
    "manager_memer",
    "reporter_qwenvl",
)


def split_episode_indices(
    episode_indices_by_task: dict[str, list[int]],
    *,
    test_ratio: float,
    seed: int,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Return deterministic, episode-disjoint train and test assignments."""
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be greater than 0 and less than 1")

    train_split: dict[str, list[int]] = {}
    test_split: dict[str, list[int]] = {}
    for task_name in sorted(episode_indices_by_task):
        candidates = sorted(set(episode_indices_by_task[task_name]))
        test_episode_count = math.ceil(len(candidates) * test_ratio)
        if len(candidates) <= test_episode_count:
            raise ValueError(
                f"Task {task_name} has {len(candidates)} candidate episodes, so "
                f"test_ratio={test_ratio} cannot leave at least one training "
                "episode."
            )

        # A task-specific RNG makes a task's split independent of which other
        # tasks happen to be selected in the same invocation.
        task_rng = random.Random(f"{seed}:{task_name}")
        selected_test = set(task_rng.sample(candidates, test_episode_count))
        test_split[task_name] = sorted(selected_test)
        train_split[task_name] = [
            episode_idx
            for episode_idx in candidates
            if episode_idx not in selected_test
        ]

    return train_split, test_split


def discover_episode_indices(
    raw_data_path: Path,
    *,
    task_names: list[str] | None,
    max_episodes: int | None,
) -> tuple[dict[str, list[int]], dict[str, str]]:
    """Read candidate episode IDs from each selected HDF5 file."""
    if not raw_data_path.is_dir():
        raise NotADirectoryError(f"Raw data directory not found: {raw_data_path}")

    requested_tasks = set(task_names) if task_names else None
    episode_indices_by_task: dict[str, list[int]] = {}
    source_files: dict[str, str] = {}
    for h5_path in sorted(raw_data_path.glob("*.h5")):
        task_name = get_env_id_from_filename(h5_path.name)
        if requested_tasks is not None and task_name not in requested_tasks:
            continue
        if task_name in episode_indices_by_task:
            raise ValueError(
                f"Multiple HDF5 files resolve to task {task_name!r}: "
                f"{source_files[task_name]} and {h5_path}"
            )
        with h5py.File(h5_path, "r") as data:
            episode_indices_by_task[task_name] = get_episode_indices(
                data, max_episodes
            )
        source_files[task_name] = str(h5_path.resolve())

    if requested_tasks is not None:
        missing_tasks = requested_tasks - episode_indices_by_task.keys()
        if missing_tasks:
            raise ValueError(
                f"No HDF5 file found for requested task(s): {sorted(missing_tasks)}"
            )
    if not episode_indices_by_task:
        raise ValueError(f"No selected .h5 files found directly under {raw_data_path}")
    return episode_indices_by_task, source_files


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build trainset/testset outputs by randomly holding out complete "
            "RoboMME episodes."
        )
    )
    parser.add_argument(
        "--dataset_type",
        choices=DATASET_TYPES,
        default="manager_qwenvl",
    )
    parser.add_argument(
        "--raw_data_path",
        type=Path,
        default=Path("data/robomme_h5_data"),
    )
    parser.add_argument(
        "--preprocessed_data_path",
        type=Path,
        default=Path("data/trinity_preprocessed_data/manager_data"),
        help="Parent output path; defaults are <path>/trainset and <path>/testset.",
    )
    parser.add_argument(
        "--train_preprocessed_data_path",
        type=Path,
        default=None,
        help="Custom train output root instead of <preprocessed_data_path>/trainset.",
    )
    parser.add_argument(
        "--test_preprocessed_data_path",
        type=Path,
        default=None,
        help="Custom test output root instead of <preprocessed_data_path>/testset.",
    )
    parser.add_argument(
        "--split_manifest_path",
        type=Path,
        default=None,
        help="JSON split record; defaults to <preprocessed_data_path>/split_manifest.json.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of complete episodes selected for testing per task (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random split seed (default: 42).",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Consider only the first N sorted episodes per task before splitting.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Only build these RoboMME task names, e.g. --tasks BinFill PickXtimes.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Write the same visualization outputs as the selected existing builder.",
    )
    return parser.parse_args()


def _validate_output_paths(
    *,
    raw_data_path: Path,
    train_output_path: Path,
    test_output_path: Path,
) -> None:
    raw = raw_data_path.resolve()
    train = train_output_path.resolve()
    test = test_output_path.resolve()
    if train == test or train in test.parents or test in train.parents:
        raise ValueError(
            "Train and test output roots must be distinct, non-nested directories: "
            f"train={train}, test={test}"
        )
    for name, output in (("train", train), ("test", test)):
        if output == raw or output in raw.parents:
            raise ValueError(
                f"The {name} output path must not equal or contain the raw data "
                f"directory: output={output}, raw={raw}"
            )


def _build_one_split(
    *,
    dataset_type: str,
    raw_data_path: Path,
    output_path: Path,
    task_names: list[str],
    episode_indices_by_task: dict[str, list[int]],
    visualize: bool,
    duplicate_samples: bool,
) -> None:
    common_kwargs: dict[str, Any] = {
        "raw_data_path": str(raw_data_path),
        "preprocessed_data_path": str(output_path),
        "visualize": visualize,
        "task_names": task_names,
        "episode_indices_by_task": episode_indices_by_task,
    }
    if dataset_type == "executer":
        processor = ExecuterDatasetProcessor(**common_kwargs)
    elif dataset_type == "manager_qwenvl":
        processor = QwenVLManagerDatasetBuilder(
            **common_kwargs,
            manager_dir_name="qwenvl",
            duplicate_samples=duplicate_samples,
        )
    elif dataset_type == "manager_memer":
        processor = MemerManagerDatasetBuilder(
            **common_kwargs,
            manager_dir_name="memer",
            duplicate_samples=duplicate_samples,
        )
    elif dataset_type == "manager_qpa":
        processor = QPAManagerDatasetBuilder(
            **common_kwargs,
            manager_dir_name=QPA_DIR_NAME,
            duplicate_samples=duplicate_samples,
        )
    elif dataset_type == "reporter_qwenvl":
        processor = ReporterDatasetBuilder(
            **common_kwargs,
            reporter_dir_name="reporter_qwenvl",
            duplicate_samples=duplicate_samples,
        )
    else:  # pragma: no cover - argparse prevents this branch.
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    processor.run()


def main() -> None:
    args = _parse_args()
    started_at = time.perf_counter()
    base_output_path = args.preprocessed_data_path.expanduser().resolve()
    raw_data_path = args.raw_data_path.expanduser().resolve()
    train_output_path = (
        args.train_preprocessed_data_path.expanduser().resolve()
        if args.train_preprocessed_data_path is not None
        else base_output_path / "trainset"
    )
    test_output_path = (
        args.test_preprocessed_data_path.expanduser().resolve()
        if args.test_preprocessed_data_path is not None
        else base_output_path / "testset"
    )
    manifest_path = (
        args.split_manifest_path.expanduser().resolve()
        if args.split_manifest_path is not None
        else base_output_path / "split_manifest.json"
    )
    _validate_output_paths(
        raw_data_path=raw_data_path,
        train_output_path=train_output_path,
        test_output_path=test_output_path,
    )

    candidates, source_files = discover_episode_indices(
        raw_data_path,
        task_names=args.tasks,
        max_episodes=args.max_episodes,
    )
    train_split, test_split = split_episode_indices(
        candidates,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    selected_tasks = sorted(candidates)

    manifest = {
        "dataset_type": args.dataset_type,
        "raw_data_path": str(raw_data_path),
        "train_preprocessed_data_path": str(train_output_path),
        "test_preprocessed_data_path": str(test_output_path),
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "max_episodes_per_task": args.max_episodes,
        "train_duplicate_samples": True,
        "test_duplicate_samples": False,
        "tasks": {
            task_name: {
                "source_file": source_files[task_name],
                "all_episode_ids": candidates[task_name],
                "train_episode_ids": train_split[task_name],
                "test_episode_ids": test_split[task_name],
            }
            for task_name in selected_tasks
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Split manifest: {manifest_path}")
    for task_name in selected_tasks:
        print(
            f"{task_name}: train={len(train_split[task_name])} episodes, "
            f"test={len(test_split[task_name])} episodes, "
            f"test IDs={test_split[task_name]}"
        )

    print(f"\nBuilding trainset at {train_output_path}")
    _build_one_split(
        dataset_type=args.dataset_type,
        raw_data_path=raw_data_path,
        output_path=train_output_path,
        task_names=selected_tasks,
        episode_indices_by_task=train_split,
        visualize=args.visualize,
        duplicate_samples=True,
    )
    print(f"\nBuilding testset at {test_output_path}")
    _build_one_split(
        dataset_type=args.dataset_type,
        raw_data_path=raw_data_path,
        output_path=test_output_path,
        task_names=selected_tasks,
        episode_indices_by_task=test_split,
        visualize=args.visualize,
        duplicate_samples=False,
    )
    print(f"Time taken: {(time.perf_counter() - started_at) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
