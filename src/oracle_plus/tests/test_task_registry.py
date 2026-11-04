from __future__ import annotations

from types import SimpleNamespace

import pytest

from oracle_plus.action_runtime import StopCubeActionAdapter
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
        (
            "PickXtimes",
            "pick up the green cube for the second time",
            "solve_pickup",
        ),
        (
            "PickXtimes",
            "place the green cube onto the target",
            "solve_putonto_whenhold",
        ),
        ("PickXtimes", "press the button to stop", "solve_button"),
        ("SwingXtimes", "pick up the red cube", "solve_pickup"),
        (
            "SwingXtimes",
            "move to the top of the right-side target for the first time",
            "solve_swingonto_whenhold",
        ),
        (
            "SwingXtimes",
            "put the red cube on the table",
            "solve_putdown_whenhold",
        ),
        ("SwingXtimes", "press the button", "solve_button"),
        (
            "StopCube",
            "move to the top of the button to prepare",
            "solve_button_ready",
        ),
        ("StopCube", "remain static", "solve_hold_obj_absTimestep"),
        (
            "StopCube",
            "press the button to stop the cube on the target",
            "solve_button",
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
        "PickXtimes": ("solve_pickup",),
        "SwingXtimes": ("solve_pickup", "solve_swingonto_whenhold"),
        "StopCube": (),
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


@pytest.mark.parametrize(
    "task",
    ["PickXtimes", "SwingXtimes", "StopCube"],
)
@pytest.mark.parametrize("include_description", [True, False])
@pytest.mark.parametrize("include_wrist", [True, False])
def test_counting_task_prompt_variants_build_matching_image_placeholders(
    task: str,
    include_description: bool,
    include_wrist: bool,
) -> None:
    messages = build_messages(
        task=task,
        call_type="followup",
        task_goal="test counting goal",
        include_description=include_description,
        include_wrist=include_wrist,
        history_subgoals=["previous subgoal"],
        current_subgoal="remain static" if task == "StopCube" else "current subgoal",
        observation_count=3,
        overall_image_shape=(256, 256),
    )
    placeholder_count = sum(
        message["content"].count("<image>") for message in messages
    )
    views = 2 if include_wrist else 1
    assert placeholder_count == (3 + 1) * views


def test_stop_cube_wait_uses_incremental_official_checkpoints() -> None:
    base = SimpleNamespace(steps_press=450, interval=30)
    env = SimpleNamespace(unwrapped=base, elapsed_steps=0)
    runtime = SimpleNamespace(env=env, planner=object())
    adapter = StopCubeActionAdapter(runtime)
    targets: list[int] = []

    def solve_hold(fake_env, _planner, **kwargs) -> None:
        target = kwargs["absTimestep"]
        targets.append(target)
        fake_env.elapsed_steps = target

    for _ in range(5):
        adapter._solve_incremental_wait(solve_hold)  # noqa: SLF001

    assert targets == [100, 200, 300, 400, 420]

    # A new episode has a smaller elapsed step and must restart checkpoint state.
    env.elapsed_steps = 0
    adapter._solve_incremental_wait(solve_hold)  # noqa: SLF001
    assert targets[-1] == 100


def test_stop_cube_prompt_marks_each_wait_checkpoint_complete() -> None:
    prompt = get_prompt(
        "StopCube",
        include_description=True,
        include_wrist=False,
    )
    assert 'completed=true with\nnext_subgoal="remain static"' in prompt.system_prompt
    assert "completed=false" not in prompt.system_prompt
