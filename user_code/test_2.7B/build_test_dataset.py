"""Build fixed-episode Qwen 2.7B feasibility-test datasets from RoboMME HDF5.

The builder creates ``subgoal_action_test.jsonl``. Every response contains the
completion decision, next subgoal, and matching BinFill MPLib wrapper name.

Unlike ``scripts/build_trainset_testset.py``, this script samples exactly N
episodes per selected task and does not calculate a percentage split.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from prompts import available_tasks
from prompts import build_messages
from prompts import get_prompt
from prompts import match_action_name

_DEPENDENCY_ERROR: ImportError | None = None
try:
    import h5py
    import imageio.v3 as iio
    import numpy as np
except ImportError as error:  # Let ``--help`` work outside the RoboMME env.
    _DEPENDENCY_ERROR = error

if TYPE_CHECKING:
    import h5py

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "test_2.7B"


@dataclass(frozen=True)
class Segment:
    start_step: int
    end_step: int
    subgoal: str
    next_subgoal: str


def _directory_name(value: str) -> str:
    name = value.strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise argparse.ArgumentTypeError(
            "name must be one non-empty directory name without slashes"
        )
    return name


def _get_env_id(dataset: h5py.File, filename: str) -> str:
    task = dataset.attrs.get("task")
    if isinstance(task, np.ndarray):
        task = task.reshape(-1)[0] if task.size else None
    if isinstance(task, (bytes, np.bytes_)):
        task = task.decode("utf-8")
    if task is not None and str(task).strip():
        return str(task).strip()
    stem = Path(filename).stem.removesuffix("_with_failure")
    return stem.split("_")[-1]


def _get_episode_indices(dataset: h5py.File) -> list[int]:
    return sorted(
        int(key.split("_", 1)[1])
        for key in dataset.keys()
        if key.startswith("episode_")
    )


def _get_timestep_indices(episode: h5py.Group) -> list[int]:
    return sorted(
        int(key.rsplit("_", 1)[1])
        for key in episode.keys()
        if key.startswith("timestep_")
    )


def _get_task_goal(episode: h5py.Group) -> str:
    return _decode_text(episode["setup"]["task_goal"][()]).strip()


def _decode_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0] if value.size else ""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _first_execution_step(episode: h5py.Group, steps: list[int]) -> int:
    for step in steps:
        info = episode[f"timestep_{step}"].get("info")
        if info is None or "is_video_demo" not in info:
            return step
        if not bool(info["is_video_demo"][()]):
            return step
    raise ValueError("Episode contains no execution timestep")


def _effective_subgoal(episode: h5py.Group, step: int, last: str | None) -> str:
    info = episode[f"timestep_{step}"]["info"]
    raw = _decode_text(info["simple_subgoal"][()]).strip()
    if "complete" in raw.casefold() and last:
        return last
    return raw


def _build_segments(episode: h5py.Group) -> list[Segment]:
    all_steps = _get_timestep_indices(episode)
    first_step = _first_execution_step(episode, all_steps)
    execution_steps = [step for step in all_steps if step >= first_step]
    effective_by_step: dict[int, str] = {}
    text_change_starts: list[tuple[int, str]] = []
    recorded_boundary_starts: list[tuple[int, str]] = []
    last: str | None = None
    for step in execution_steps:
        subgoal = _effective_subgoal(episode, step, last)
        effective_by_step[step] = subgoal
        if subgoal != last:
            text_change_starts.append((step, subgoal))
        info = episode[f"timestep_{step}"]["info"]
        if "is_subgoal_boundary" in info and bool(info["is_subgoal_boundary"][()]):
            recorded_boundary_starts.append((step, subgoal))
        last = subgoal

    # The explicit boundary flag is authoritative and preserves consecutive
    # repeated instructions (for example, moving above several targets). Older
    # datasets without this field fall back to simple-subgoal text changes.
    if recorded_boundary_starts:
        starts = recorded_boundary_starts
        if starts[0][0] != first_step:
            starts.insert(0, (first_step, effective_by_step[first_step]))
    else:
        starts = text_change_starts
    if not starts:
        raise ValueError("Episode contains no simple-subgoal segment")

    final_step = execution_steps[-1]
    segments: list[Segment] = []
    for index, (start_step, subgoal) in enumerate(starts):
        if index + 1 < len(starts):
            end_step, next_subgoal = starts[index + 1]
        else:
            end_step, next_subgoal = final_step, "STOP"
        if end_step > start_step:
            segments.append(
                Segment(
                    start_step=start_step,
                    end_step=end_step,
                    subgoal=subgoal,
                    next_subgoal=next_subgoal,
                )
            )
    return segments


def _sample_steps(start: int, end: int, stride: int) -> list[int]:
    distance = end - start
    if distance <= 0:
        return []
    interval_count = max(1, math.ceil(distance / stride))
    sampled = [
        int(start + distance * index / interval_count)
        for index in range(interval_count + 1)
    ]
    return sorted({int(step) for step in sampled if int(step) > start})


def _write_view(
    *,
    episode: h5py.Group,
    task: str,
    episode_id: int,
    step: int,
    view: Literal["overall", "wrist"],
    images_root: Path,
) -> str:
    dataset_key = "front_rgb" if view == "overall" else "wrist_rgb"
    observation = episode[f"timestep_{step}"]["obs"]
    if dataset_key not in observation:
        raise KeyError(
            f"{task} episode_{episode_id} timestep_{step} has no {dataset_key!r}"
        )
    view_dir = images_root / view
    view_dir.mkdir(parents=True, exist_ok=True)
    image_path = view_dir / f"{task}_ep{episode_id}_step{step}_{view}.png"
    if not image_path.exists():
        iio.imwrite(image_path, observation[dataset_key][()])
    return str(image_path.resolve())


def _paired_paths(
    episode: h5py.Group,
    task: str,
    episode_id: int,
    step: int,
    images_root: Path,
) -> tuple[str, str]:
    return (
        _write_view(
            episode=episode,
            task=task,
            episode_id=episode_id,
            step=step,
            view="overall",
            images_root=images_root,
        ),
        _write_view(
            episode=episode,
            task=task,
            episode_id=episode_id,
            step=step,
            view="wrist",
            images_root=images_root,
        ),
    )


def _finalize_row(
    *,
    messages: list[dict[str, str]],
    images: list[str],
    expected: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    messages = [
        *messages,
        {
            "role": "assistant",
            "content": json.dumps(expected, ensure_ascii=False),
        },
    ]
    actual_placeholders = sum(
        message["content"].count("<image>")
        for message in messages
        if message["role"] != "assistant"
    )
    if actual_placeholders != len(images):
        raise RuntimeError(
            f"Prompt/image mismatch: {actual_placeholders} placeholders for "
            f"{len(images)} images"
        )
    return {
        "messages": messages,
        "images": images,
        "metadata": metadata,
    }


def _initial_row(
    *,
    task: str,
    episode_id: int,
    task_goal: str,
    segments: list[Segment],
    initial_paths: tuple[str, str],
) -> dict[str, Any]:
    if not segments:
        raise ValueError("Initial prompt requires at least one subgoal segment")
    prompt = get_prompt(task)
    subgoal_sequence = [segment.subgoal for segment in segments]
    action_sequence: list[str] = []
    for subgoal in subgoal_sequence:
        action_name = match_action_name(task, subgoal)
        if action_name is None:
            raise ValueError(
                f"No action-catalogue match for {task} episode_{episode_id}: "
                f"{subgoal!r}"
            )
        action_sequence.append(action_name)
    expected: dict[str, Any] = {
        "subgoal_sequence": subgoal_sequence,
        "action_sequence": action_sequence,
        "completed": False,
        "next_subgoal": subgoal_sequence[0],
        "action_name": action_sequence[0],
    }
    target_subgoal = subgoal_sequence[0]

    messages = build_messages(
        task=task,
        call_type="initial",
        task_goal=task_goal,
    )
    return _finalize_row(
        messages=messages,
        images=list(initial_paths),
        expected=expected,
        metadata={
            "mode": "subgoal_action",
            "call_type": "initial",
            "prompt_version": prompt.version,
            "prompt_sha256": prompt.content_hash,
            "task": task,
            "episode": episode_id,
            "segment_index": 0,
            "init_step": segments[0].start_step,
            "current_step": segments[0].start_step,
            "observation_steps": [],
            "current_subgoal": None,
            "target_subgoal": target_subgoal,
            "completed": False,
        },
    )


def _row(
    *,
    task: str,
    episode_id: int,
    task_goal: str,
    history_subgoals: list[str],
    segment_index: int,
    segment: Segment,
    current_step: int,
    observation_steps: list[int],
    init_paths: tuple[str, str],
    observation_paths: list[tuple[str, str]],
) -> dict[str, Any]:
    completed = current_step == segment.end_step
    target_subgoal = segment.next_subgoal if completed else segment.subgoal
    prompt = get_prompt(task)
    description = (
        "The current subgoal is complete."
        if completed
        else "The current subgoal is not complete."
    )
    if target_subgoal == "STOP":
        action_name = None
    else:
        action_name = match_action_name(task, target_subgoal)
        if action_name is None:
            raise ValueError(
                f"No action-catalogue match for {task} episode_{episode_id} "
                f"step {current_step}: {target_subgoal!r}"
            )
    expected: dict[str, Any] = {
        "description": description,
        "completed": completed,
        "next_subgoal": target_subgoal,
        "action_name": action_name,
    }

    messages = build_messages(
        task=task,
        call_type="followup",
        task_goal=task_goal,
        history_subgoals=history_subgoals,
        current_subgoal=segment.subgoal,
        observation_count=len(observation_paths),
    )
    images = [*init_paths]
    for pair in observation_paths:
        images.extend(pair)
    return _finalize_row(
        messages=messages,
        images=images,
        expected=expected,
        metadata={
            "mode": "subgoal_action",
            "call_type": "followup",
            "prompt_version": prompt.version,
            "prompt_sha256": prompt.content_hash,
            "task": task,
            "episode": episode_id,
            "segment_index": segment_index,
            "init_step": segment.start_step,
            "current_step": current_step,
            "observation_steps": observation_steps,
            "current_subgoal": segment.subgoal,
            "target_subgoal": target_subgoal,
            "completed": completed,
        },
    )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")


def _discover_sources(raw_data_path: Path, tasks: set[str] | None) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for h5_path in sorted(raw_data_path.glob("*.h5")):
        with h5py.File(h5_path, "r") as dataset:
            task = _get_env_id(dataset, h5_path.name)
        if tasks is not None and task not in tasks:
            continue
        if task in sources:
            raise ValueError(f"Multiple HDF5 files resolve to task {task!r}")
        sources[task] = h5_path
    if tasks is not None:
        missing = tasks - sources.keys()
        if missing:
            raise ValueError(f"No HDF5 source found for task(s): {sorted(missing)}")
    if not sources:
        raise ValueError(f"No selected HDF5 files found under {raw_data_path}")
    return sources


def _select_episodes(
    *,
    available: list[int],
    episode_count: int,
    seed: int,
    task: str,
    explicit_ids: list[int] | None,
) -> list[int]:
    if explicit_ids is not None:
        missing = sorted(set(explicit_ids) - set(available))
        if missing:
            raise ValueError(f"{task} does not contain episode IDs {missing}")
        return sorted(set(explicit_ids))
    if len(available) < episode_count:
        raise ValueError(
            f"{task} has only {len(available)} episodes, fewer than requested "
            f"episode_count={episode_count}"
        )
    rng = random.Random(f"{seed}:{task}")
    return sorted(rng.sample(available, episode_count))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-data-path",
        type=Path,
        default=REPO_ROOT / "data" / "robomme_data_h5",
        help="Directory containing RoboMME .h5 files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Dataset parent directory (default: data/test_2.7B).",
    )
    parser.add_argument(
        "--dataset-name",
        type=_directory_name,
        default="binfill_test",
        help="Dataset directory name under --output-root (default: binfill_test).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["BinFill"],
        help="Tasks to build (default: BinFill). A task-specific prompt module is required.",
    )
    parser.add_argument(
        "--episode-count",
        type=int,
        default=10,
        help="Exact number of randomly selected episodes per task (default: 10).",
    )
    parser.add_argument(
        "--episode-ids",
        nargs="+",
        type=int,
        default=None,
        help="Explicit IDs instead of --episode-count; requires exactly one selected task.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if _DEPENDENCY_ERROR is not None:
        raise RuntimeError(
            "Dataset construction requires h5py, imageio, and numpy. Activate "
            "the RoboMME server environment used by scripts/build_dataset.py."
        ) from _DEPENDENCY_ERROR
    if args.episode_count < 1:
        raise ValueError("--episode-count must be at least 1")
    if args.history_size < 1:
        raise ValueError("--history-size must be at least 1")
    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    raw_data_path = args.raw_data_path.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_dir = output_root / args.dataset_name
    if not raw_data_path.is_dir():
        raise NotADirectoryError(raw_data_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose a new directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    images_root = output_dir / "images"

    selected_task_names = set(args.tasks) if args.tasks else None
    unsupported_tasks = (selected_task_names or set()) - set(available_tasks())
    if unsupported_tasks:
        raise ValueError(
            f"No task-specific prompts for {sorted(unsupported_tasks)}; "
            f"available tasks: {available_tasks()}"
        )
    sources = _discover_sources(raw_data_path, selected_task_names)
    if args.episode_ids is not None and len(sources) != 1:
        raise ValueError("--episode-ids requires exactly one selected task/HDF5 source")

    output_path = output_dir / "subgoal_action_test.jsonl"
    row_count = 0
    selected_by_task: dict[str, list[int]] = {}

    for task, source_path in sorted(sources.items()):
        with h5py.File(source_path, "r") as dataset:
            selected_episodes = _select_episodes(
                available=_get_episode_indices(dataset),
                episode_count=args.episode_count,
                seed=args.seed,
                task=task,
                explicit_ids=args.episode_ids,
            )
            selected_by_task[task] = selected_episodes
            for episode_id in selected_episodes:
                print(f"Processing {task} episode_{episode_id}")
                episode = dataset[f"episode_{episode_id}"]
                task_goal = _get_task_goal(episode)
                segments = _build_segments(episode)
                initial_paths = _paired_paths(
                    episode,
                    task,
                    episode_id,
                    segments[0].start_step,
                    images_root,
                )
                initial_row = _initial_row(
                    task=task,
                    episode_id=episode_id,
                    task_goal=task_goal,
                    segments=segments,
                    initial_paths=initial_paths,
                )
                _append_jsonl(output_path, initial_row)
                row_count += 1
                for segment_index, segment in enumerate(segments):
                    init_paths = _paired_paths(
                        episode,
                        task,
                        episode_id,
                        segment.start_step,
                        images_root,
                    )
                    history_pairs: deque[tuple[int, tuple[str, str]]] = deque(
                        maxlen=args.history_size
                    )
                    for current_step in _sample_steps(
                        segment.start_step, segment.end_step, args.stride
                    ):
                        current_paths = _paired_paths(
                            episode,
                            task,
                            episode_id,
                            current_step,
                            images_root,
                        )
                        history_pairs.append((current_step, current_paths))
                        observation_steps = [item[0] for item in history_pairs]
                        observation_paths = [item[1] for item in history_pairs]
                        row = _row(
                            task=task,
                            episode_id=episode_id,
                            task_goal=task_goal,
                            history_subgoals=[
                                previous.subgoal
                                for previous in segments[:segment_index]
                            ],
                            segment_index=segment_index,
                            segment=segment,
                            current_step=current_step,
                            observation_steps=observation_steps,
                            init_paths=init_paths,
                            observation_paths=observation_paths,
                        )
                        _append_jsonl(output_path, row)
                        row_count += 1

    manifest = {
        "raw_data_path": str(raw_data_path),
        "dataset_name": args.dataset_name,
        "output_root": str(output_root),
        "output_dir": str(output_dir),
        "episode_selection": "explicit" if args.episode_ids is not None else "fixed_count_random",
        "episode_count_per_task": None if args.episode_ids is not None else args.episode_count,
        "seed": args.seed,
        "history_size": args.history_size,
        "stride": args.stride,
        "selected_episode_ids": selected_by_task,
        "sources": {task: str(path.resolve()) for task, path in sources.items()},
        "prompts": {
            task: get_prompt(task).metadata()
            for task in selected_by_task
        },
        "rows": row_count,
        "output": str(output_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

