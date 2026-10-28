"""Create a temporary timing summary from completed episode JSON files."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_TIMING_DIR = (
    REPO_ROOT
    / "runs/evaluation/symbolic-simple-subgoal/trinity_v2.2.2"
    / "seed0/repeat1/BinFill/timing_logs"
)
COMPONENTS = ("framework", "reporter", "manager", "executer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timing-dir",
        type=Path,
        default=DEFAULT_TIMING_DIR,
        help="Directory containing <task>_ep<N>.json timing files.",
    )
    parser.add_argument(
        "--task",
        default="BinFill",
        help="Filename task prefix and expected JSON task (default: BinFill).",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=30,
        help="Maximum completed episodes to include (default: 30).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path; default: "
            "<timing-dir>/temporary_summary_<N>episodes.json."
        ),
    )
    return parser.parse_args()


def _episode_id(path: Path, pattern: re.Pattern[str]) -> int:
    match = pattern.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected timing filename: {path.name}")
    return int(match.group("episode"))


def _load_episode(path: Path, task: str, episode_id: int) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if payload.get("task") != task:
        raise ValueError(
            f"Task mismatch in {path}: expected {task!r}, "
            f"got {payload.get('task')!r}"
        )
    if int(payload.get("episode_id", -1)) != episode_id:
        raise ValueError(
            f"Episode mismatch in {path}: filename={episode_id}, "
            f"JSON={payload.get('episode_id')!r}"
        )
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Missing summary object in {path}")
    for component in COMPONENTS:
        component_summary = summary.get(component)
        if not isinstance(component_summary, dict):
            raise ValueError(f"Missing summary.{component} in {path}")
        try:
            call_count = int(component_summary["call_count"])
            total_ms = float(component_summary["total_ms"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid summary.{component} call_count/total_ms in {path}"
            ) from error
        if call_count < 0 or total_ms < 0:
            raise ValueError(f"Negative timing value for {component} in {path}")
    return payload


def summarize(
    timing_dir: Path,
    *,
    task: str,
    num_episodes: int,
) -> dict[str, Any]:
    if num_episodes < 1:
        raise ValueError("--num-episodes must be at least 1")
    timing_dir = timing_dir.expanduser().resolve()
    if not timing_dir.is_dir():
        raise FileNotFoundError(f"Timing directory does not exist: {timing_dir}")

    pattern = re.compile(rf"{re.escape(task)}_ep(?P<episode>\d+)\.json")
    candidates = sorted(
        (
            path
            for path in timing_dir.iterdir()
            if path.is_file() and pattern.fullmatch(path.name)
        ),
        key=lambda path: _episode_id(path, pattern),
    )
    selected = candidates[:num_episodes]
    if not selected:
        raise FileNotFoundError(
            f"No {task}_ep<N>.json files found in {timing_dir}"
        )

    aggregate = {
        component: {"call_count": 0, "total_ms": 0.0}
        for component in COMPONENTS
    }
    episode_details: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    for path in selected:
        episode_id = _episode_id(path, pattern)
        payload = _load_episode(path, task, episode_id)
        status = str(payload.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        episode_details.append(
            {
                "episode_id": episode_id,
                "status": status,
                "source": str(path),
            }
        )
        for component in COMPONENTS:
            values = payload["summary"][component]
            aggregate[component]["call_count"] += int(values["call_count"])
            aggregate[component]["total_ms"] += float(values["total_ms"])

    summary: dict[str, dict[str, int | float | None]] = {}
    for component in COMPONENTS:
        call_count = int(aggregate[component]["call_count"])
        total_ms = float(aggregate[component]["total_ms"])
        average_ms = total_ms / call_count if call_count else None
        summary[component] = {
            "call_count": call_count,
            "average_ms": round(average_ms, 3) if average_ms is not None else None,
            "average_seconds": (
                round(average_ms / 1000, 3) if average_ms is not None else None
            ),
            "total_ms": round(total_ms, 3),
            "total_seconds": round(total_ms / 1000, 3),
        }

    included_count = len(selected)
    return {
        "task": task,
        "status": "temporary_partial_summary",
        "unit": "ms",
        "source_directory": str(timing_dir),
        "requested_episode_count": num_episodes,
        "available_episode_count": len(candidates),
        "included_episode_count": included_count,
        "is_requested_count_complete": included_count == num_episodes,
        "selection": "lowest numeric episode IDs among completed timing files",
        "aggregation": (
            "Weighted across component calls: sum(total_ms) / sum(call_count)."
        ),
        "episode_ids": [detail["episode_id"] for detail in episode_details],
        "status_counts": status_counts,
        "summary": summary,
        "episodes": episode_details,
    }


def main() -> None:
    args = parse_args()
    result = summarize(
        args.timing_dir,
        task=args.task,
        num_episodes=args.num_episodes,
    )
    timing_dir = args.timing_dir.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else timing_dir / f"temporary_summary_{args.num_episodes}episodes.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)

    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(
        f"Included {result['included_episode_count']}/"
        f"{result['requested_episode_count']} requested episodes"
    )
    print(f"Temporary summary: {output_path}")


if __name__ == "__main__":
    main()
