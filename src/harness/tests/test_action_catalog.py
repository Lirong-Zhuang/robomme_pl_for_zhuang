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
        "solve_hold_abs_timestep",
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


def _load_actions() -> list[dict]:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    assert catalog["schema_version"] == "action_catalog.v2"
    return catalog["actions"]


def test_action_catalog_entries_are_unique_and_described() -> None:
    actions = _load_actions()
    names = [action["name"] for action in actions]

    assert len(names) == len(set(names))
    base_fields = {"name", "description", "perception"}
    implementation_fields = {"implementation_status", "implementation_note"}
    optional_fields = implementation_fields | {"semantic_arguments"}
    assert all(
        base_fields <= set(action)
        and set(action) <= base_fields | optional_fields
        for action in actions
    )
    assert all(action["name"].strip() for action in actions)
    assert all(action["description"].strip() for action in actions)
    for action in actions:
        if "implementation_status" not in action:
            assert "implementation_note" not in action
        else:
            assert action["implementation_status"] in {"incomplete"}
            assert action["implementation_note"].strip()

        semantic_arguments = action.get("semantic_arguments", [])
        assert isinstance(semantic_arguments, list)
        names = [argument["name"] for argument in semantic_arguments]
        assert len(names) == len(set(names))
        for argument in semantic_arguments:
            expected = {"name", "source", "allowed_values"}
            if "runtime_mapping" in argument:
                expected.add("runtime_mapping")
            assert set(argument) == expected
            assert argument["source"] == "subgoal_text_tokens"
            assert argument["allowed_values"]
            if "runtime_mapping" in argument:
                assert set(argument["runtime_mapping"]) == set(
                    argument["allowed_values"]
                )


def test_stopcube_wait_action_is_explicitly_deferred() -> None:
    actions = {action["name"]: action for action in _load_actions()}
    wait_action = actions["solve_hold_abs_timestep"]

    assert wait_action["implementation_status"] == "incomplete"
    assert "absolute_timestep" in wait_action["implementation_note"]
    assert "StopCube" in wait_action["implementation_note"]


def test_perception_contracts_limit_vlm_to_2d_point_hints() -> None:
    allowed_modes = {"none", "vlm_points", "scene_target_set"}
    allowed_color_policies = {"required", "optional", "not_used", "fixed"}
    allowed_f2g_outputs = {"pose", "size"}
    vlm_backends = {
        "grasp_grounding",
        "side_grasp_grounding",
        "location_grounding",
    }

    for action in _load_actions():
        perception = action["perception"]
        assert set(perception) == {"mode", "targets"}
        assert perception["mode"] in allowed_modes
        assert isinstance(perception["targets"], list)

        if perception["mode"] == "none":
            assert perception["targets"] == []
            continue

        assert perception["targets"]
        roles = [target["role"] for target in perception["targets"]]
        assert len(roles) == len(set(roles))
        for target in perception["targets"]:
            common_fields = {
                "role",
                "point_required",
                "color_hint",
                "grounding_backend",
                "point_semantics",
            }
            if target["color_hint"] == "fixed":
                common_fields.add("fixed_color")
            if "allowed_colors" in target:
                common_fields.add("allowed_colors")
            backend = target["grounding_backend"]
            if backend == "location_grounding":
                expected_fields = common_fields | {"geometry_outputs"}
            else:
                expected_fields = common_fields | {
                    "segmentation_profile",
                    "f2g_outputs",
                }
            assert set(target) == expected_fields
            assert target["role"].strip()
            assert isinstance(target["point_required"], bool)
            assert target["color_hint"] in allowed_color_policies
            if target["color_hint"] == "fixed":
                assert target["fixed_color"].strip()
            if "allowed_colors" in target:
                assert target["color_hint"] == "required"
                assert target["allowed_colors"]
                assert len(target["allowed_colors"]) == len(
                    set(target["allowed_colors"])
                )
            assert target["point_semantics"].strip()
            if perception["mode"] == "vlm_points":
                assert backend in vlm_backends
                assert any(
                    word in target["point_semantics"].casefold()
                    for word in ("center", "centroid")
                )
            else:
                assert backend == "scene_grounding"
            if backend == "location_grounding":
                assert target["geometry_outputs"] == ["position"]
            else:
                assert target["segmentation_profile"].strip()
                assert set(target["f2g_outputs"]) <= allowed_f2g_outputs
                assert "pose" in target["f2g_outputs"]

        if perception["mode"] == "vlm_points":
            assert all(target["point_required"] for target in perception["targets"])
        else:
            assert all(
                not target["point_required"] for target in perception["targets"]
            )


def test_only_grasps_use_vlm_points() -> None:
    expected_roles = {
        "solve_pickup": {"object"},
        "solve_put_into_bin": set(),
        "solve_button": set(),
        "grasp_and_lift_peg": {"peg"},
        "grasp_and_lift_peg_side": {"peg"},
        "solve_push_to_target_with_peg": set(),
        "solve_push_to_target": set(),
        "solve_putonto_target": set(),
        "solve_putdown": set(),
        "solve_button_ready": set(),
        "solve_hold_abs_timestep": set(),
        "solve_pickup_bin": {"object"},
        "solve_swingonto_whenhold": set(),
        "insert_peg": set(),
        "solve_pattern_direction": set(),
        "solve_swingonto_with_direction": set(),
    }

    actions = _load_actions()
    assert {action["name"] for action in actions} == set(expected_roles)
    for action in actions:
        roles = {target["role"] for target in action["perception"]["targets"]}
        assert roles == expected_roles[action["name"]]
        assert action["perception"]["mode"] == ("vlm_points" if roles else "none")


def test_peg_endpoints_require_black_or_white() -> None:
    actions = {action["name"]: action for action in _load_actions()}

    peg, = actions["grasp_and_lift_peg_side"]["perception"]["targets"]
    assert peg["allowed_colors"] == ["black", "white"]


def test_catalog_covers_every_non_example_task_oracle_action() -> None:
    available = {action["name"] for action in _load_actions()}

    assert len(REMAINING_TASK_ACTIONS) == 13
    for task_name, required in REMAINING_TASK_ACTIONS.items():
        missing = required - available
        assert not missing, f"{task_name} is missing action candidates: {missing}"
