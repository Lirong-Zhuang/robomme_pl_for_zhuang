from __future__ import annotations

from types import SimpleNamespace

import pytest

from oracle_plus.action_runtime import StopCubeActionAdapter
from oracle_plus.action_runtime import StopCubeReachEdgeGate
from oracle_plus.action_runtime import SwingVisitEdgeGate
from oracle_plus.action_runtime import build_action_adapter
from oracle_plus.prompts import available_tasks
from oracle_plus.prompts import build_messages
from oracle_plus.prompts import get_point_action_names
from oracle_plus.prompts import get_prompt
from oracle_plus.prompts import resolve_action_name
from oracle_plus.vlm import SubgoalDecision


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
        (
            "SwingXtimes",
            "release the green cube and leave it resting on the tabletop",
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
            (
                "press the button now because the cube is currently approaching "
                "the target for the third time"
            ),
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


def test_stop_cube_rejects_legacy_conditional_press_wording() -> None:
    with pytest.raises(ValueError):
        resolve_action_name(
            "StopCube",
            "as the cube approaches the target for the third time, press the button",
        )


def test_descriptive_stop_form_is_terminal() -> None:
    stop_text = "STOP, only after the button press completes"

    assert resolve_action_name("SwingXtimes", stop_text) is None
    assert SubgoalDecision(
        completed=True,
        next_subgoal=stop_text,
    ).is_stop


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


def test_stop_cube_environment_timing_toggle_defaults_on_and_can_be_disabled() -> None:
    assert build_action_adapter("StopCube", object()).use_env_timing is True
    adapter = build_action_adapter(
        "StopCube",
        object(),
        stopcube_use_env_timing=False,
    )
    assert adapter.use_env_timing is False


def test_swing_putdown_release_status_uses_grasp_and_pose_state() -> None:
    cube = SimpleNamespace(pose=SimpleNamespace(p=[[0.0, 0.0, 0.03]]))
    agent = SimpleNamespace(
        tcp=SimpleNamespace(pose=SimpleNamespace(p=[[0.0, 0.0, 0.15]])),
        is_grasping=lambda _cube: False,
    )
    base = SimpleNamespace(target_cube=cube, agent=agent)
    adapter = build_action_adapter(
        "SwingXtimes",
        SimpleNamespace(env=SimpleNamespace(unwrapped=base)),
    )

    assert adapter.putdown_release_status() == {
        "is_grasping": False,
        "cube_z": 0.03,
        "gripper_z": 0.15,
        "released_on_tabletop": True,
    }

    agent.is_grasping = lambda _cube: True
    assert adapter.putdown_release_status()["released_on_tabletop"] is False


@pytest.mark.parametrize(
    ("subgoal", "side"),
    [
        (
            "move to the top of the robot-right target (camera-left disk) "
            "for the third time",
            "right",
        ),
        (
            "move to the top of the robot-left target (camera-right disk) "
            "for the third time",
            "left",
        ),
        ("move to the top of the right-side target", "right"),
        ("move to the top of the left-side target", "left"),
    ],
)
def test_swing_semantic_target_side_ignores_camera_frame_words(
    subgoal: str,
    side: str,
) -> None:
    adapter = build_action_adapter("SwingXtimes", object())
    assert adapter.semantic_target_side(subgoal) == side


def test_swing_visit_edge_counts_only_a_new_expected_disk_entry() -> None:
    gate = SwingVisitEdgeGate()

    gate.sync("right")
    first_left = gate.accept(expected_side="left", occupied_side="left")
    assert first_left["accepted"] is True
    assert first_left["left_visits"] == 1

    gate.sync("left")
    repeated_left = gate.accept(expected_side="left", occupied_side="left")
    assert repeated_left["accepted"] is False
    assert repeated_left["left_visits"] == 1

    gate.sync("left")
    second_right = gate.accept(expected_side="right", occupied_side="right")
    assert second_right["accepted"] is True
    assert second_right["right_visits"] == 1


def test_swing_visit_edge_accepts_initial_expected_occupancy_once() -> None:
    gate = SwingVisitEdgeGate()

    gate.sync("right")
    first_right = gate.accept(expected_side="right", occupied_side="right")
    assert first_right["accepted"] is True
    assert first_right["initial_expected_occupancy"] is True

    gate.sync("right")
    repeated_right = gate.accept(expected_side="right", occupied_side="right")
    assert repeated_right["accepted"] is False
    assert repeated_right["right_visits"] == 1


def test_swing_visit_status_uses_fixed_disk_x_order_and_bounded_regions() -> None:
    cube = SimpleNamespace(name="cube")
    target_right = SimpleNamespace(name="target_right")
    target_left = SimpleNamespace(name="target_left")
    base = SimpleNamespace(
        target_cube=cube,
        target_right=target_right,
        target_left=target_left,
    )
    pixels = {
        "cube": [168, 92],
        "target_right": [80, 88],
        "target_left": [170, 88],
    }

    class Runtime:
        env = SimpleNamespace(unwrapped=base)

        @staticmethod
        def project_actor_center(actor, *, projection_z=None):
            return {
                "name": actor.name,
                "position": [0.0, 0.0, 0.0],
                "projected_pixel": pixels[actor.name],
            }

    adapter = build_action_adapter("SwingXtimes", Runtime())
    status = adapter.visit_status()
    assert status["occupied_side"] == "left"
    assert status["targets"]["right"]["projected_pixel"][0] < (
        status["targets"]["left"]["projected_pixel"][0]
    )

    pixels["cube"] = [125, 88]
    assert adapter.visit_status()["occupied_side"] is None


def test_swing_visit_status_projects_held_cube_at_target_plane() -> None:
    cube = SimpleNamespace(name="cube")
    target_right = SimpleNamespace(name="target_right")
    target_left = SimpleNamespace(name="target_left")
    base = SimpleNamespace(
        target_cube=cube,
        target_right=target_right,
        target_left=target_left,
    )

    class Runtime:
        env = SimpleNamespace(unwrapped=base)

        @staticmethod
        def project_actor_center(actor, *, projection_z=None):
            if actor is cube:
                assert projection_z == pytest.approx(0.005)
                return {
                    "name": actor.name,
                    "position": [-0.1179, -0.1358, 0.1018],
                    "projection_position": [-0.1179, -0.1358, projection_z],
                    "projected_pixel": [94, 82],
                }
            if actor is target_right:
                return {
                    "name": actor.name,
                    "position": [-0.1154, -0.1413, 0.005],
                    "projected_pixel": [94, 82],
                }
            return {
                "name": actor.name,
                "position": [-0.0488, 0.1615, 0.005],
                "projected_pixel": [169, 94],
            }

    status = build_action_adapter("SwingXtimes", Runtime()).visit_status()

    assert status["occupied_side"] == "right"
    assert status["cube"]["position"][2] == pytest.approx(0.1018)
    assert status["cube"]["projection_position"][2] == pytest.approx(0.005)
    assert status["distance_pixels"]["right"] == 0.0


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


def test_stop_cube_wait_caps_at_environment_press_boundary_by_default() -> None:
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
    assert adapter.wait_progress_available() is False


def test_stop_cube_wait_can_ignore_environment_timing() -> None:
    base = SimpleNamespace()
    env = SimpleNamespace(unwrapped=base, elapsed_steps=410)
    runtime = SimpleNamespace(env=env, planner=object())
    adapter = StopCubeActionAdapter(runtime, use_env_timing=False)
    targets: list[int] = []

    def solve_hold(fake_env, _planner, **kwargs) -> None:
        target = kwargs["absTimestep"]
        targets.append(target)
        fake_env.elapsed_steps = target

    adapter._solve_incremental_wait(solve_hold)  # noqa: SLF001
    adapter._solve_incremental_wait(solve_hold)  # noqa: SLF001

    assert targets == [430, 450]
    with pytest.raises(RuntimeError, match="environment timing is disabled"):
        adapter.wait_progress_available()


def test_stop_cube_timing_reports_environment_values() -> None:
    base = SimpleNamespace(
        steps_press=450,
        interval=30,
        move_interval=120,
        stop_time=4,
    )
    env = SimpleNamespace(unwrapped=base, elapsed_steps=419)
    adapter = StopCubeActionAdapter(SimpleNamespace(env=env, planner=object()))

    assert adapter.wait_timing() == {
        "elapsed_steps": 419,
        "steps_press": 450,
        "interval": 30,
        "final_wait_target": 420,
        "move_interval": 120,
        "stop_time": 4,
    }


def test_stop_cube_reach_edge_requires_a_physical_exit_before_retrigger() -> None:
    gate = StopCubeReachEdgeGate()

    gate.sync(False)
    first_entry = gate.observe(True)
    assert first_entry["entered"] is True
    assert first_entry["observed_visits"] == 1
    assert gate.accept_report(True) is True

    # Remaining inside cannot create or consume another visit, even if the VLM
    # briefly reports false and then true again.
    assert gate.observe(True)["entered"] is False
    assert gate.accept_report(False) is False
    assert gate.accept_report(True) is False

    assert gate.observe(False)["exited"] is True
    second_entry = gate.observe(True)
    assert second_entry["entered"] is True
    assert second_entry["observed_visits"] == 2
    assert gate.accept_report(True) is True


def test_stop_cube_target_occupancy_uses_world_xy_region() -> None:
    cube = SimpleNamespace(pose=SimpleNamespace(p=[[0.02, 0.0, 0.02]]))
    target = SimpleNamespace(pose=SimpleNamespace(p=[[0.0, 0.0, 0.005]]))
    base = SimpleNamespace(cube=cube, target=target, cube_half_size=0.01)
    adapter = StopCubeActionAdapter(
        SimpleNamespace(
            env=SimpleNamespace(unwrapped=base),
            planner=object(),
        )
    )

    status = adapter.target_occupancy_status()
    assert status["inside"] is True
    assert status["distance_xy"] == pytest.approx(0.02)
    assert status["threshold"] == pytest.approx(0.03)

    cube.pose.p = [[0.04, 0.0, 0.02]]
    assert adapter.target_occupancy_status()["inside"] is False


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
        "The ordinal in the press subgoal must exactly equal N" in prompt.system_prompt
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
    assert "Any\ntarget occupancy already visible there has already been consumed" in (
        prompt.system_prompt
    )
    assert "require it\nto visibly leave and then re-enter" in prompt.system_prompt
    assert "A continuous\noccupancy spanning subgoal-init" in prompt.system_prompt
    assert "skip every numbered remain-static subgoal whose" in prompt.system_prompt
    assert "the first follow-up may go directly" not in prompt.system_prompt
    assert "the cube is now moving away, do not output the press subgoal" in (
        prompt.system_prompt
    )
    assert "the third approach is non-final and must not trigger a press" in (
        prompt.system_prompt
    )
    assert "visibly moving\ntoward the target but is still short of its center" in (
        prompt.system_prompt
    )
    assert "If the task goal requests stopping on the third" not in (
        prompt.system_prompt
    )
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
def test_swing_prompt_uses_color_independent_visible_target_visit_rule(
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
    assert "Do not use any target color or\ncolor change" in prompt.system_prompt
    assert "smaller\nimage x-coordinate is always the robot-right target" in (
        prompt.system_prompt
    )
    assert "Never infer target side from the arm direction, wrist\nimage" in (
        prompt.system_prompt
    )
    assert "in the newest overall observation, the center of the held cube" in (
        prompt.system_prompt
    )
    assert "only the cube's stable occupancy of the disk is shown" in (
        prompt.system_prompt
    )
    assert "over the target only in an older frame does not complete" in (
        prompt.system_prompt
    )
    assert "red highlight" not in prompt.system_prompt.casefold()
    assert "currently red" not in prompt.system_prompt.casefold()
    assert "Both target disks are part of the tabletop" in prompt.system_prompt
    assert "Do not\nrequire the cube to move away from the final target disk" in (
        prompt.system_prompt
    )
    assert "Do not repeat release merely because the cube remains on a\ntarget disk" in (
        prompt.system_prompt
    )
    assert "release the [red/blue/green] cube and leave it resting" in (
        prompt.system_prompt
    )
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
