#!/usr/bin/env python3
"""Export gripper observations around Reporter-critical ``pick up`` true steps.

For Reporter dataset generation, a subgoal becomes true at the timestep where
the recorded simple subgoal changes to the next subgoal. This script reproduces
that transition definition, finds every ``pick up`` span in one HDF5 episode,
and exports a configurable window around each critical-true timestep.

Edit the USER CONFIG section, then run from the repository root:

    python user_code/inspect_pickup_gripper_window.py

The CSV records both observed gripper fingers, the first-finger position used
by the live Reporter gripper gate, its endpoint/intermediate classification,
and any available gripper commands from joint/eef/waypoint actions.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np


# ============================= USER CONFIG =============================
REPO_ROOT = Path(__file__).resolve().parents[1]
H5_PATH = REPO_ROOT / "data/h5_data/merged/record_dataset_BinFill_with_failure.h5"
EPISODE_NAME = "episode_0"

# Number of recorded timesteps before/after each critical true to export.
STEPS_BEFORE = 100
STEPS_AFTER = 100

# None exports every pick-up event. Set 0, 1, ... to select one event.
PICKUP_EVENT_INDEX: int | None = None
OUTPUT_DIR = REPO_ROOT / "runs/pickup_gripper_windows"

# Must match the live gate configuration in scripts/eval.sh.
GRIPPER_CLOSED_POSITION = 0.0
GRIPPER_OPEN_POSITION = 0.04
GRIPPER_ENDPOINT_BUFFER = 0.002
# ======================================================================


SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mme_vla_suite.reporter_gripper_tool import PickUpGripperGate


@dataclass(frozen=True)
class PickUpEvent:
    """One recorded pick-up subgoal and its Reporter true boundary."""

    event_index: int
    subgoal: str
    start_step: int
    critical_true_step: int


def _decode_text(value: Any) -> str:
    array = np.asarray(value)
    if array.size == 0:
        return ""
    value = array.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _read_optional_bool(group: h5py.Group, key: str) -> bool | None:
    if key not in group:
        return None
    value = np.asarray(group[key][()]).reshape(-1)
    return bool(value[0]) if value.size else None


def _read_action_gripper(timestep: h5py.Group, key: str) -> float | None:
    action = timestep.get("action")
    if action is None or key not in action:
        return None
    values = np.asarray(action[key][()], dtype=np.float32).reshape(-1)
    return float(values[-1]) if values.size else None


def _timestep_indices(episode: h5py.Group) -> list[int]:
    return sorted(
        int(key.rsplit("_", 1)[-1])
        for key in episode.keys()
        if key.startswith("timestep_")
    )


def _execution_steps(episode: h5py.Group) -> list[int]:
    """Exclude conditioning-video timesteps from subgoal transitions."""
    steps = []
    for step in _timestep_indices(episode):
        info = episode[f"timestep_{step}"]["info"]
        if _read_optional_bool(info, "is_video_demo") is not True:
            steps.append(step)
    return steps


def _effective_subgoals(
    episode: h5py.Group,
    steps: list[int],
) -> dict[int, str]:
    """Resolve the recorded ``complete`` sentinel like the dataset builder."""
    effective: dict[int, str] = {}
    last_subgoal: str | None = None
    for step in steps:
        info = episode[f"timestep_{step}"]["info"]
        raw = _decode_text(info["simple_subgoal"][()]).strip().lower()
        subgoal = last_subgoal if "complete" in raw and last_subgoal else raw
        effective[step] = subgoal
        last_subgoal = subgoal
    return effective


def find_pick_up_events(episode: h5py.Group) -> tuple[list[int], dict[int, str], list[PickUpEvent]]:
    """Find pick-up spans using the same transition boundary as Reporter data."""
    steps = _execution_steps(episode)
    if not steps:
        raise ValueError("Episode contains no execution timesteps")
    subgoals = _effective_subgoals(episode, steps)

    transitions: list[int] = []
    previous: str | None = None
    for step in steps:
        if subgoals[step] != previous:
            transitions.append(step)
            previous = subgoals[step]
    if transitions[-1] != steps[-1]:
        transitions.append(steps[-1])

    events: list[PickUpEvent] = []
    for start_step, critical_true_step in zip(transitions[:-1], transitions[1:]):
        subgoal = subgoals[start_step]
        if "pick up" not in subgoal.casefold():
            continue
        events.append(
            PickUpEvent(
                event_index=len(events),
                subgoal=subgoal,
                start_step=start_step,
                critical_true_step=critical_true_step,
            )
        )
    return steps, subgoals, events


def _gate_classification(gate: PickUpGripperGate, position: float) -> str:
    result = gate.verify_reporter_true(
        subgoal="pick up",
        gripper_position=position,
    )
    if result.passed:
        return "intermediate_pass"
    if "closed" in result.reason:
        return "closed_endpoint_reject"
    if "open" in result.reason:
        return "open_endpoint_reject"
    return result.reason


def build_rows(
    episode: h5py.Group,
    steps: list[int],
    effective_subgoals: dict[int, str],
    events: list[PickUpEvent],
    gate: PickUpGripperGate,
) -> list[dict[str, Any]]:
    """Build one combined table with a separate indexed window per event."""
    step_positions = {step: position for position, step in enumerate(steps)}
    rows: list[dict[str, Any]] = []
    for event in events:
        center = step_positions[event.critical_true_step]
        window_start = max(0, center - STEPS_BEFORE)
        window_end = min(len(steps), center + STEPS_AFTER + 1)
        for position in range(window_start, window_end):
            step = steps[position]
            timestep = episode[f"timestep_{step}"]
            info = timestep["info"]
            gripper = np.asarray(
                timestep["obs"]["gripper_state"][()],
                dtype=np.float32,
            ).reshape(-1)
            if gripper.size == 0:
                first_finger = float("nan")
                second_finger = float("nan")
            else:
                first_finger = float(gripper[0])
                second_finger = float(gripper[1]) if gripper.size > 1 else float("nan")

            rows.append(
                {
                    "event_index": event.event_index,
                    "event_subgoal": event.subgoal,
                    "event_start_step": event.start_step,
                    "critical_true_step": event.critical_true_step,
                    "is_critical_true": step == event.critical_true_step,
                    "recorded_step_offset": position - center,
                    "timestep": step,
                    "effective_simple_subgoal": effective_subgoals[step],
                    "raw_simple_subgoal": _decode_text(
                        info["simple_subgoal"][()]
                    ),
                    "is_subgoal_boundary": _read_optional_bool(
                        info,
                        "is_subgoal_boundary",
                    ),
                    "is_gripper_close": _read_optional_bool(
                        timestep["obs"],
                        "is_gripper_close",
                    ),
                    "gripper_finger_0": first_finger,
                    "gripper_finger_1": second_finger,
                    "gripper_total_opening": float(np.sum(gripper)),
                    "live_gate_position": first_finger,
                    "live_gate_classification": _gate_classification(
                        gate,
                        first_finger,
                    ),
                    "joint_action_gripper": _read_action_gripper(
                        timestep,
                        "joint_action",
                    ),
                    "eef_action_gripper": _read_action_gripper(
                        timestep,
                        "eef_action",
                    ),
                    "waypoint_action_gripper": _read_action_gripper(
                        timestep,
                        "waypoint_action",
                    ),
                }
            )
    return rows


def _select_events(events: list[PickUpEvent]) -> list[PickUpEvent]:
    if PICKUP_EVENT_INDEX is None:
        return events
    if PICKUP_EVENT_INDEX < 0 or PICKUP_EVENT_INDEX >= len(events):
        raise IndexError(
            f"PICKUP_EVENT_INDEX={PICKUP_EVENT_INDEX} is out of range; "
            f"episode has {len(events)} pick-up events"
        )
    return [events[PICKUP_EVENT_INDEX]]


def export_pick_up_windows() -> Path:
    h5_path = H5_PATH.expanduser().resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(
            f"HDF5 file not found: {h5_path}. Edit H5_PATH in the USER CONFIG section."
        )
    if STEPS_BEFORE < 0 or STEPS_AFTER < 0:
        raise ValueError("STEPS_BEFORE and STEPS_AFTER must be non-negative")

    gate = PickUpGripperGate(
        closed_position=GRIPPER_CLOSED_POSITION,
        open_position=GRIPPER_OPEN_POSITION,
        endpoint_buffer=GRIPPER_ENDPOINT_BUFFER,
    )
    with h5py.File(h5_path, "r") as h5_file:
        if EPISODE_NAME not in h5_file:
            available = sorted(
                key for key in h5_file.keys() if key.startswith("episode_")
            )
            raise KeyError(
                f"{EPISODE_NAME!r} not found. Available episodes: {available[:20]}"
            )
        episode = h5_file[EPISODE_NAME]
        steps, effective_subgoals, all_events = find_pick_up_events(episode)
        if not all_events:
            raise ValueError(
                f"{EPISODE_NAME!r} contains no simple subgoal with 'pick up'"
            )
        selected_events = _select_events(all_events)
        rows = build_rows(
            episode,
            steps,
            effective_subgoals,
            selected_events,
            gate,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / (
        f"{h5_path.stem}_{EPISODE_NAME}_pickup_gripper_windows.csv"
    )
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"HDF5: {h5_path}")
    print(f"Episode: {EPISODE_NAME}")
    print(
        "Gate interval: "
        f"({gate.accepted_lower_bound:.6f}, {gate.accepted_upper_bound:.6f})"
    )
    for event in selected_events:
        print(
            f"Pick-up event {event.event_index}: start={event.start_step}, "
            f"critical_true={event.critical_true_step}, subgoal={event.subgoal!r}"
        )
    print(f"Rows: {len(rows)}")
    print(f"CSV: {output_path}")
    return output_path


if __name__ == "__main__":
    export_pick_up_windows()
