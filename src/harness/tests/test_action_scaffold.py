from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.actions.executor import ActionExecutor
from harness.actions.env_targets import bin_position
from harness.actions.env_targets import button_position
from harness.actions.execution_state import insert_peg_obj_flag_from_grasped_endpoint
from harness.actions.execution_state import move_cube_obj_flag_from_grasped_endpoint
from harness.actions.execution_state import remember_move_cube_center_peg_grasp
from harness.actions.execution_state import remember_grasped_peg_endpoint_color
from harness.actions.execution_state import require_grasped_peg_endpoint_color
from harness.actions.oracle_reference import ORACLE_ACTION_REFERENCES
from harness.actions.push import direction_from_grounded_positions
from harness.actions.registry import CatalogAction
from harness.actions.registry import TargetBinding
from harness.actions.registry import build_action_registry
from harness.actions.types import ActionRequest
from harness.actions.types import ObjectEstimate
from harness.actions.types import Pose3D
from harness.actions.types import Size3D


CATALOG_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "action_candidates"
    / "action_catalog.json"
)


def _estimate(*, frame: str = "world", label: str | None = None) -> ObjectEstimate:
    return ObjectEstimate(
        object_id="cube-0",
        pose=Pose3D(
            position_xyz=(0.1, 0.2, 0.03),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            frame=frame,
        ),
        size=Size3D((0.04, 0.04, 0.04)),
        label=label,
    )


def test_registry_exactly_matches_action_catalog() -> None:
    catalog = json.loads(CATALOG_PATH.read_text())
    catalog_names = {entry["name"] for entry in catalog["actions"]}

    assert set(build_action_registry()) == catalog_names
    assert set(ORACLE_ACTION_REFERENCES) == catalog_names
    assert all(callable(action) for action in build_action_registry().values())


def test_pose_normalizes_quaternion() -> None:
    pose = Pose3D((0, 0, 0), (2, 0, 0, 0))

    assert pose.quaternion_wxyz == (1.0, 0.0, 0.0, 0.0)


def test_pickup_binding_unpacks_pose_and_size() -> None:
    estimate = _estimate()
    action = build_action_registry()["solve_pickup"]

    kwargs = action.bind(
        ActionRequest("solve_pickup", targets={"object": estimate})
    )

    assert kwargs == {"obj_pose": estimate.pose, "obj_size": estimate.size}


def test_pickup_validates_required_target() -> None:
    action = build_action_registry()["solve_pickup"]
    with pytest.raises(ValueError, match="missing target role"):
        action.bind(ActionRequest("solve_pickup"))


def test_planning_only_arguments_are_not_forwarded_to_action() -> None:
    estimate = _estimate()
    action = build_action_registry()["solve_pickup"]

    kwargs = action.bind(
        ActionRequest(
            "solve_pickup",
            targets={"object": estimate},
            arguments={"ordinal": "first", "color": "red"},
        )
    )

    assert kwargs == {"obj_pose": estimate.pose, "obj_size": estimate.size}


def test_binding_rejects_camera_frame_target() -> None:
    action = build_action_registry()["solve_pickup"]
    request = ActionRequest(
        "solve_pickup",
        targets={"object": _estimate(frame="front_camera")},
    )

    with pytest.raises(ValueError, match="world/base frame"):
        action.bind(request)


def test_bin_position_reads_board_with_hole_from_environment() -> None:
    pose = type("Pose", (), {"p": [[0.1, 0.2, 0.3]]})()
    actor = type("Actor", (), {"pose": pose})()
    base = type("BinFill", (), {"board_with_hole": actor})()
    env = type("Wrapper", (), {"unwrapped": base})()

    assert bin_position(env) == pytest.approx([0.1, 0.2, 0.3])


def test_button_position_matches_oracle_plus_task_specific_preference() -> None:
    regular_pose = type("Pose", (), {"p": [[0.1, 0.2, 0.3]]})()
    left_pose = type("Pose", (), {"p": [[0.4, 0.5, 0.6]]})()
    regular = type("Actor", (), {"pose": regular_pose})()
    left = type("Actor", (), {"pose": left_pose})()
    base = type(
        "ButtonUnmaskSwap",
        (),
        {"button": regular, "button_left": left},
    )()
    env = type("Wrapper", (), {"unwrapped": base})()

    assert button_position(env) == pytest.approx([0.4, 0.5, 0.6])


def test_button_position_selects_two_button_env_by_semantic_ordinal() -> None:
    left_pose = type("Pose", (), {"p": [[0.1, 0.2, 0.3]]})()
    right_pose = type("Pose", (), {"p": [[0.4, 0.5, 0.6]]})()
    left = type("Actor", (), {"pose": left_pose})()
    right = type("Actor", (), {"pose": right_pose})()
    base = type(
        "ButtonUnmaskSwap",
        (),
        {"button_left": left, "button_right": right},
    )()

    assert button_position(base, ordinal="first") == pytest.approx([0.1, 0.2, 0.3])
    assert button_position(base, ordinal="second") == pytest.approx([0.4, 0.5, 0.6])


def test_every_catalog_action_binds_its_explicit_parameters() -> None:
    estimate = _estimate()
    black_peg = _estimate(label="black")
    requests = {
        "solve_pickup": ActionRequest("solve_pickup", targets={"object": estimate}),
        "solve_put_into_bin": ActionRequest("solve_put_into_bin"),
        "solve_button": ActionRequest("solve_button"),
        "grasp_and_lift_peg": ActionRequest(
            "grasp_and_lift_peg", targets={"peg": estimate}
        ),
        "grasp_and_lift_peg_side": ActionRequest("grasp_and_lift_peg_side", targets={"peg": black_peg}),
        "solve_push_to_target_with_peg": ActionRequest(
            "solve_push_to_target_with_peg",
            targets={"object": estimate, "target": estimate},
        ),
        "solve_push_to_target": ActionRequest(
            "solve_push_to_target",
            targets={"object": estimate, "target": estimate},
        ),
        "solve_putonto_target": ActionRequest("solve_putonto_target", targets={"target": estimate}),
        "solve_putdown": ActionRequest("solve_putdown"),
        "solve_button_ready": ActionRequest("solve_button_ready"),
        "solve_hold_abs_timestep": ActionRequest(
            "solve_hold_abs_timestep",
            arguments={"absolute_timestep": 100},
        ),
        "solve_pickup_bin": ActionRequest("solve_pickup_bin", targets={"object": estimate}),
        "solve_swingonto_whenhold": ActionRequest("solve_swingonto_whenhold", targets={"target": estimate}),
        "insert_peg": ActionRequest(
            "insert_peg",
            targets={"peg": estimate, "box": estimate},
            arguments={"direction": 1},
        ),
        "solve_pattern_direction": ActionRequest(
            "solve_pattern_direction",
            target_sets={"targets": (estimate,)},
            arguments={"direction": "forward"},
        ),
        "solve_swingonto_with_direction": ActionRequest(
            "solve_swingonto_with_direction",
            target_sets={"targets": (estimate,)},
            arguments={"side": "left", "direction": "clockwise"},
        ),
    }
    registry = build_action_registry()

    assert set(requests) == set(registry)
    for name, request in requests.items():
        registry[name].bind(request)


def test_peg_push_direction_comes_from_grounded_world_positions() -> None:
    cube = Pose3D((0.0, 0.2, 0.03), (1.0, 0.0, 0.0, 0.0))
    target = Pose3D((0.1, -0.1, 0.01), (1.0, 0.0, 0.0, 0.0))

    assert direction_from_grounded_positions(cube, target) == 1
    assert direction_from_grounded_positions(target, cube) == -1


def test_peg_endpoint_color_maps_to_task_specific_oracle_flags() -> None:
    base = SimpleNamespace()
    env = SimpleNamespace(unwrapped=base)

    remember_grasped_peg_endpoint_color(env, "black")
    assert require_grasped_peg_endpoint_color(env) == "black"
    assert move_cube_obj_flag_from_grasped_endpoint(env) == 1
    assert insert_peg_obj_flag_from_grasped_endpoint(env) == -1

    remember_grasped_peg_endpoint_color(env, "white")
    assert require_grasped_peg_endpoint_color(env) == "white"
    assert move_cube_obj_flag_from_grasped_endpoint(env) == -1
    assert insert_peg_obj_flag_from_grasped_endpoint(env) == 1

    remember_move_cube_center_peg_grasp(env)
    assert move_cube_obj_flag_from_grasped_endpoint(env) == 1
    with pytest.raises(RuntimeError, match="black or white endpoint"):
        insert_peg_obj_flag_from_grasped_endpoint(env)


def test_executor_calls_function_inside_oracle_runtime() -> None:
    calls = []
    estimate = _estimate()

    def solve_done(env, planner, *, obj_pose, obj_size):
        calls.append(("solve", env, planner, obj_pose, obj_size))
        return "raw planner result"

    action = CatalogAction(
        "done",
        solve_done,
        targets=(TargetBinding("object", "obj_pose", "obj_size"),),
    )

    class FakeRuntime:
        env = object()
        planner = object()

        def execute(self, solve):
            calls.append(("runtime.execute",))
            assert solve() == "raw planner result"
            return SimpleNamespace(
                status="success",
                stopped=True,
                simulator_frame_count=7,
                info={"collector": "oracle_plus"},
            )

    result = ActionExecutor(FakeRuntime(), {"done": action}).execute(
        ActionRequest("done", targets={"object": estimate})
    )

    assert calls == [
        ("runtime.execute",),
        ("solve", FakeRuntime.env, FakeRuntime.planner, estimate.pose, estimate.size),
    ]
    assert result.status == "success"
    assert result.simulator_frame_count == 7
    assert result.info == {"collector": "oracle_plus"}
