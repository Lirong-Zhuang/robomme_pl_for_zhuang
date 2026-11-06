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
            "move to the top of the robot-right target (camera-left disk) for the first time",
            "solve_swingonto_whenhold",
        ),
        (
            "SwingXtimes",
            "move to the top of the robot-left target (camera-right disk) for the first time",
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
        (
            "StopCube",
            "remain static while the cube reaches the target for the second time",
            "solve_hold_obj_absTimestep",
        ),
        (
            "StopCube",
            "as the cube approaches the target for the third time, press the button",
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
        current_subgoal=(
            "remain static while the cube reaches the target for the first time"
            if task == "StopCube"
            else "current subgoal"
        ),
        observation_count=3,
        overall_image_shape=(256, 256),
    )
    placeholder_count = sum(
        message["content"].count("<image>") for message in messages
    )
    views = 2 if include_wrist else 1
    assert placeholder_count == (3 + 1) * views


def test_stop_cube_wait_advances_in_twenty_step_chunks() -> None:
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

    assert targets == [20, 40, 60, 80, 100]

    # A reset naturally starts another sequence from the new current timestep.
    env.elapsed_steps = 0
    adapter._solve_incremental_wait(solve_hold)  # noqa: SLF001
    assert targets[-1] == 20


def test_stop_cube_wait_caps_at_final_target_and_then_makes_no_call() -> None:
    base = SimpleNamespace(steps_press=450, interval=30)
    env = SimpleNamespace(unwrapped=base, elapsed_steps=410)
    runtime = SimpleNamespace(env=env, planner=object())
    adapter = StopCubeActionAdapter(runtime)
    targets: list[int] = []

    def solve_hold(fake_env, _planner, **kwargs) -> None:
        target = kwargs["absTimestep"]
        targets.append(target)
        fake_env.elapsed_steps = target

    adapter._solve_incremental_wait(solve_hold)  # noqa: SLF001
    adapter._solve_incremental_wait(solve_hold)  # noqa: SLF001

    assert targets == [420]


def test_stop_cube_wait_progress_is_unavailable_only_at_final_boundary() -> None:
    base = SimpleNamespace(
        steps_press=450,
        interval=30,
        move_interval=120,
        stop_time=4,
    )
    env = SimpleNamespace(unwrapped=base, elapsed_steps=419)
    adapter = StopCubeActionAdapter(SimpleNamespace(env=env, planner=object()))

    assert adapter.wait_progress_available() is True
    assert adapter.wait_timing() == {
        "elapsed_steps": 419,
        "steps_press": 450,
        "interval": 30,
        "final_wait_target": 420,
        "move_interval": 120,
        "stop_time": 4,
    }

    env.elapsed_steps = 420
    assert adapter.wait_progress_available() is False


@pytest.mark.parametrize("include_description", [True, False])
@pytest.mark.parametrize("include_wrist", [True, False])
def test_stop_cube_prompt_uses_numbered_waits_in_all_calls(
    include_description: bool,
    include_wrist: bool,
) -> None:
    prompt = get_prompt(
        "StopCube",
        include_description=include_description,
        include_wrist=include_wrist,
    )
    assert (
        "remain static while the cube reaches the target for the [ordinal] time"
        in prompt.system_prompt
    )
    assert (
        "remain static while the cube reaches the target for the first time"
        in prompt.system_prompt
    )
    assert (
        "remain static while the cube reaches the target for the second time"
        in prompt.system_prompt
    )
    assert (
        "as the cube approaches the target for the third time, press the button"
        in prompt.system_prompt
    )
    assert "Never output the bare or shortened form \"remain static\"" in (
        prompt.system_prompt
    )
    assert "return completed=false and repeat the exact same full numbered" in (
        prompt.system_prompt
    )
    assert "Target visits observed while moving above the button to prepare still count" in (
        prompt.system_prompt
    )
    assert "skip every numbered remain-static subgoal whose" in prompt.system_prompt
    assert "the first follow-up may go directly" in prompt.system_prompt
    assert 'next_subgoal="remain static"' not in prompt.system_prompt


@pytest.mark.parametrize("include_description", [True, False])
@pytest.mark.parametrize("include_wrist", [True, False])
def test_pick_xtimes_prompt_uses_liberal_pickup_completion(
    include_description: bool,
    include_wrist: bool,
) -> None:
    prompt = get_prompt(
        "PickXtimes",
        include_description=include_description,
        include_wrist=include_wrist,
    )
    assert "Approximate visual evidence is sufficient" in prompt.system_prompt
    assert "Do not require a large or clearly measurable lifting height" in (
        prompt.system_prompt
    )
    assert "cube moving upward together with the gripper" in prompt.system_prompt


@pytest.mark.parametrize("include_description", [True, False])
@pytest.mark.parametrize("include_wrist", [True, False])
def test_swing_prompt_uses_original_visible_target_visit_rule(
    include_description: bool,
    include_wrist: bool,
) -> None:
    prompt = get_prompt(
        "SwingXtimes",
        include_description=include_description,
        include_wrist=include_wrist,
    )
    assert "Count a target visit only when the held cube visibly reaches" in (
        prompt.system_prompt
    )
    assert "Either cue alone is sufficient" not in prompt.system_prompt
    assert "Approximate visual evidence is sufficient" not in prompt.system_prompt
    assert "focus primarily on the wrist images" not in prompt.system_prompt
    assert (
        "robot-right target (camera-left disk) for the [first/second/third/...] time"
        in prompt.system_prompt
    )
    assert (
        "robot-left target (camera-right disk) for the [first/second/third/...] time"
        in prompt.system_prompt
    )
    assert 'never use a bare\n"left-side" or "right-side" target/disk label' in (
        prompt.system_prompt
    )

    initial_messages = prompt.build_messages(
        call_type="initial",
        task_goal="swing once",
        observation_count=0,
        overall_image_shape=(256, 256),
    )
    followup_messages = prompt.build_messages(
        call_type="followup",
        task_goal="swing once",
        history_subgoals=[],
        current_subgoal="move to the top of the left-side target for the first time",
        observation_count=3,
        overall_image_shape=(256, 256),
    )
    for messages in (initial_messages, followup_messages):
        user_prompt = messages[-1]["content"]
        assert "Target-side reminder for this call" in user_prompt
        assert 'robot-right target as "robot-right target (camera-left disk)"' in (
            user_prompt
        )
        assert 'robot-left target as "robot-left target (camera-right disk)"' in (
            user_prompt
        )
        assert "not the wrist image, to identify left versus right" in user_prompt
        assert "never use a bare" in user_prompt
