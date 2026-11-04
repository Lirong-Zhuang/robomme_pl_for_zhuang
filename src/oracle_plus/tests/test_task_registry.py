from __future__ import annotations

import pytest

from oracle_plus.action_runtime import build_action_adapter
from oracle_plus.prompts import available_tasks
from oracle_plus.prompts import build_messages
from oracle_plus.prompts import get_point_action_names
from oracle_plus.prompts import get_prompt
from oracle_plus.prompts import resolve_action_name


@pytest.mark.parametrize(
    ("task", "subgoal", "action_name"),
    [
        ("ButtonUnmask", "press the button", "solve_button"),
        (
            "ButtonUnmask",
            "pick up the container that hides the blue cube",
            "solve_pickup_bin",
        ),
        ("ButtonUnmask", "put down the container", "solve_putdown_whenhold"),
        ("PickHighlight", "press button", "solve_button"),
        (
            "PickHighlight",
            "pick up the first highlighted cube, which is red",
            "solve_pickup",
        ),
        (
            "PickHighlight",
            "place the cube onto the table",
            "solve_putdown_whenhold",
        ),
        ("MoveCube", "pick up the peg", "grasp_and_lift_peg_side"),
        (
            "MoveCube",
            "hook the cube to the target with the peg",
            "solve_push_to_target_with_peg",
        ),
        (
            "MoveCube",
            "close gripper and push the cube to the target",
            "solve_push_to_target",
        ),
        ("MoveCube", "pick up the cube", "solve_pickup"),
        (
            "MoveCube",
            "place the cube onto the target",
            "solve_putonto_whenhold",
        ),
    ],
)
def test_subgoals_resolve_to_trusted_actions(
    task: str,
    subgoal: str,
    action_name: str,
) -> None:
    assert resolve_action_name(task, subgoal) == action_name


def test_new_tasks_have_matching_point_registries() -> None:
    expected = {
        "ButtonUnmask": ("solve_pickup_bin",),
        "PickHighlight": ("solve_pickup",),
        "MoveCube": (),
    }
    assert set(expected).issubset(available_tasks())
    for task, point_actions in expected.items():
        adapter = build_action_adapter(task, object())
        assert get_point_action_names(task) == point_actions
        assert adapter.point_actions == point_actions


@pytest.mark.parametrize("include_wrist", [True, False])
def test_move_cube_initial_prompt_aligns_with_sampled_demo_images(
    include_wrist: bool,
) -> None:
    observation_count = 10
    messages = build_messages(
        task="MoveCube",
        call_type="initial",
        task_goal="move the cube as demonstrated",
        include_description=True,
        include_wrist=include_wrist,
        observation_count=observation_count,
        overall_image_shape=(256, 256),
    )
    placeholder_count = sum(
        message["content"].count("<image>") for message in messages
    )
    expected_views = 2 if include_wrist else 1
    assert placeholder_count == observation_count * expected_views
    assert get_prompt(
        "MoveCube",
        include_description=True,
        include_wrist=include_wrist,
    ).initial_observation_mode == "demonstration"
