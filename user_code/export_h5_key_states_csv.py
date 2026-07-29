#!/usr/bin/env python3
"""Export per-timestep key-state diagnostics from one HDF5 episode.

The initial timestep and HDF5 subgoal boundaries are treated as key states.
Joint and gripper distances are measurements relative to the previous key
state; they do not participate in key-state selection.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one HDF5 episode as a per-timestep key-state CSV."
    )
    parser.add_argument("h5_path", type=Path, help="Path to the source HDF5 file")
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to export (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/h5_key_state_csv"),
        help="Directory for the generated CSV",
    )
    return parser.parse_args()


def decode_text(value: Any) -> str:
    """Decode scalar or one-element HDF5 string values."""
    array = np.asarray(value)
    if array.ndim > 0 and array.size == 1:
        value = array.reshape(-1)[0]
    elif array.ndim == 0:
        value = array.item()
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def read_bool(group: h5py.Group, key: str, default: bool = False) -> bool:
    if key not in group:
        return default
    return bool(np.asarray(group[key][()]).reshape(-1)[0])


def read_subgoal(info: h5py.Group, key: str) -> str:
    if key not in info:
        return ""
    return decode_text(info[key][()])


def get_timestep_keys(episode: h5py.Group) -> list[str]:
    return sorted(
        (key for key in episode.keys() if key.startswith("timestep_")),
        key=lambda key: int(key.rsplit("_", 1)[-1]),
    )


def get_absolute_state(timestep: h5py.Group) -> np.ndarray:
    """Return 7 observed joint values plus total two-finger opening."""
    joint_state = np.asarray(
        timestep["obs"]["joint_state"][()],
        dtype=np.float32,
    ).reshape(-1)
    gripper_state = np.asarray(
        timestep["obs"]["gripper_state"][()],
        dtype=np.float32,
    ).reshape(-1)
    if joint_state.size != 7:
        raise ValueError(
            f"Expected 7 joint values, got shape {joint_state.shape}"
        )
    gripper_opening = np.sum(gripper_state, dtype=np.float32)
    return np.concatenate(
        [joint_state, np.asarray([gripper_opening], dtype=np.float32)]
    )


def export_episode(
    h5_path: Path,
    episode_idx: int,
    output_dir: Path,
) -> Path:
    episode_key = f"episode_{episode_idx}"
    h5_path = h5_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{h5_path.stem}_{episode_key}_key_states.csv"

    with h5py.File(h5_path, "r") as h5_file:
        if episode_key not in h5_file:
            available = sorted(
                key for key in h5_file.keys() if key.startswith("episode_")
            )
            raise KeyError(
                f"{episode_key!r} not found in {h5_path}. "
                f"Available episodes: {available[:20]}"
            )

        episode = h5_file[episode_key]
        timestep_keys = get_timestep_keys(episode)
        if not timestep_keys:
            raise ValueError(f"{episode_key!r} contains no timesteps")

        fieldnames = [
            "timestep",
            "simple_subgoal",
            "is_subgoal_boundary",
            "joint_distance_to_previous_key_state",
            "gripper_distance_to_previous_key_state",
            *(f"joint_{i}" for i in range(7)),
            "gripper_opening",
        ]

        previous_key_state: np.ndarray | None = None
        rows: list[dict[str, Any]] = []

        for position, timestep_key in enumerate(timestep_keys):
            timestep_idx = int(timestep_key.rsplit("_", 1)[-1])
            timestep = episode[timestep_key]
            info = timestep["info"]
            state = get_absolute_state(timestep)
            is_boundary = read_bool(info, "is_subgoal_boundary")
            is_initial_state = position == 0

            if previous_key_state is None:
                joint_distance: float | None = None
                gripper_distance: float | None = None
            else:
                joint_distance = float(
                    np.linalg.norm(state[:7] - previous_key_state[:7])
                )
                gripper_distance = abs(
                    float(state[7] - previous_key_state[7])
                )

            is_key_state = is_initial_state or is_boundary
            row: dict[str, Any] = {
                "timestep": timestep_idx,
                "simple_subgoal": read_subgoal(info, "simple_subgoal"),
                "is_subgoal_boundary": is_boundary,
                "joint_distance_to_previous_key_state": (
                    "" if joint_distance is None else joint_distance
                ),
                "gripper_distance_to_previous_key_state": (
                    "" if gripper_distance is None else gripper_distance
                ),
                "gripper_opening": float(state[7]),
            }
            row.update({f"joint_{i}": float(state[i]) for i in range(7)})
            rows.append(row)

            if is_key_state:
                previous_key_state = state

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main() -> None:
    args = parse_args()
    output_path = export_episode(
        h5_path=args.h5_path,
        episode_idx=args.episode,
        output_dir=args.output_dir,
    )
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
