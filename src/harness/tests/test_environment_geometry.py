from types import SimpleNamespace

import pytest

from harness.actions.env_targets import bind_environment_geometry
from harness.actions.execution_state import remember_grasped_peg_endpoint_color
from harness.actions.semantic_arguments import resolve_action_arguments
from harness.actions.types import ActionRequest


def _actor(x, y=0, z=0):
    return SimpleNamespace(pose=SimpleNamespace(p=[x, y, z], q=[1, 0, 0, 0]))


@pytest.mark.parametrize("actor_name", ["goal_site", "target"])
def test_placement_reads_live_destination_pose(actor_name):
    disk = _actor(0.2)
    env = SimpleNamespace(**{actor_name: disk})
    request = ActionRequest("solve_putonto_target")
    first = bind_environment_geometry(env, request)
    assert first.targets["target"].pose.position_xyz == (0.2, 0, 0)
    disk.pose.p[0] = 0.4
    assert bind_environment_geometry(env, request).targets["target"].pose.position_xyz == (0.4, 0, 0)


def test_push_reads_execution_cube_and_disk_without_demo_geometry():
    env = SimpleNamespace(
        cube=_actor(0.1), goal_site=_actor(0.2), cube_half_size=0.02,
        cube_2=_actor(0.7), goal_site_2=_actor(0.8),
    )
    bound = bind_environment_geometry(env, ActionRequest("solve_push_to_target"))
    assert bound.targets["object"].pose.position_xyz == (0.1, 0, 0)
    assert bound.targets["object"].size.xyz == (0.04, 0.04, 0.04)
    assert bound.targets["target"].pose.position_xyz == (0.2, 0, 0)


def test_swing_binds_the_side_selected_by_the_vlm():
    env = SimpleNamespace(target_left=_actor(0, 0.1), target_right=_actor(0, -0.1))
    name = "solve_swingonto_whenhold"
    for side, y in [("left", 0.1), ("right", -0.1)]:
        args = resolve_action_arguments(name, subgoal=f"swing above the robot-{side} target")
        bound = bind_environment_geometry(env, ActionRequest(name, arguments=args))
        assert bound.targets["target"].pose.position_xyz == (0, y, 0)


def test_insertion_uses_the_free_end_of_the_actually_held_peg():
    heads = [_actor(0.1), _actor(0.2)]
    tails = [_actor(0.3), _actor(0.4)]
    env = SimpleNamespace(
        peg_heads=heads, peg_tails=tails, box=_actor(0.5),
        peg_head=heads[0], peg_tail=tails[0],
        agent=SimpleNamespace(is_grasping=lambda actor: actor is tails[1]),
    )
    remember_grasped_peg_endpoint_color(env, "white")
    bound = bind_environment_geometry(env, ActionRequest("insert_peg"))
    assert bound.targets["peg"].pose.position_xyz == (0.2, 0, 0)
    assert bound.targets["box"].pose.position_xyz == (0.5, 0, 0)


def test_route_uses_all_grid_indices_without_reading_the_planned_route():
    env = SimpleNamespace(targets_grid=[_actor(i) for i in range(9)], selected_buttons=[_actor(100)])
    bound = bind_environment_geometry(env, ActionRequest("solve_swingonto_with_direction"))
    assert [item.pose.position_xyz[0] for item in bound.target_sets["targets"]] == list(range(9))
