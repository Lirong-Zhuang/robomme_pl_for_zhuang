from __future__ import annotations

import json
from pathlib import Path


CATALOG_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "action_candidates"
    / "action_catalog.json"
)


REMAINING_TASK_ACTIONS = {
    "StopCube": {
        "solve_button_ready",
        "solve_hold_obj_abs_timestep",
        "solve_button",
    },
    "PickXtimes": {
        "solve_pickup",
        "solve_putonto_target",
        "solve_button",
    },
    "SwingXtimes": {
        "solve_pickup",
        "solve_swingonto_whenhold",
        "solve_putdown",
        "solve_button",
    },
    "ButtonUnmask": {
        "solve_button",
        "solve_pickup_bin",
        "solve_putdown",
    },
    "VideoUnmask": {"solve_pickup_bin", "solve_putdown"},
    "VideoUnmaskSwap": {"solve_pickup_bin", "solve_putdown"},
    "ButtonUnmaskSwap": {
        "solve_button",
        "solve_pickup_bin",
        "solve_putdown",
    },
    "VideoRepick": {"solve_pickup", "solve_putdown", "solve_button"},
    "VideoPlaceButton": {"solve_pickup", "solve_putonto_target"},
    "VideoPlaceOrder": {"solve_pickup", "solve_putonto_target"},
    "InsertPeg": {"grasp_and_lift_peg_side", "insert_peg"},
    "PatternLock": {"solve_pattern_direction"},
    "RouteStick": {"solve_swingonto_with_direction"},
}


def _load_actions() -> list[dict[str, str]]:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    assert catalog["schema_version"] == "action_catalog.v1"
    return catalog["actions"]


def test_action_catalog_entries_are_unique_and_described() -> None:
    actions = _load_actions()
    names = [action["name"] for action in actions]

    assert len(names) == len(set(names))
    assert all(set(action) == {"name", "description"} for action in actions)
    assert all(action["name"].strip() for action in actions)
    assert all(action["description"].strip() for action in actions)


def test_catalog_covers_every_non_example_task_oracle_action() -> None:
    available = {action["name"] for action in _load_actions()}

    assert len(REMAINING_TASK_ACTIONS) == 13
    for task_name, required in REMAINING_TASK_ACTIONS.items():
        missing = required - available
        assert not missing, f"{task_name} is missing action candidates: {missing}"
