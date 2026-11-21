from __future__ import annotations

from pathlib import Path

import pytest

from harness.grounding import GroundingContractRegistry


CATALOG_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "action_candidates"
    / "action_catalog.json"
)


def _registry() -> GroundingContractRegistry:
    return GroundingContractRegistry.from_catalog(CATALOG_PATH)


def _point(role: str, *, color_hint: str | None = None) -> dict:
    return {
        "status": "ok",
        "image_ref": "current_front",
        "targets": {
            role: {
                "point_xy": [120, 90],
                "color_hint": color_hint,
            }
        },
    }


def _not_required() -> dict:
    return {"status": "not_required", "image_ref": None, "targets": {}}


def test_colored_pickup_accepts_only_2d_point_and_color() -> None:
    grounding = _registry().validate(
        "solve_pickup",
        _point("object", color_hint="red"),
        image_shape=(256, 256),
    )

    assert grounding["targets"]["object"] == {
        "point_xy": [120.0, 90.0],
        "color_hint": "red",
    }


def test_vlm_geometry_fields_are_rejected() -> None:
    grounding = _point("object", color_hint="red")
    grounding["targets"]["object"]["pose"] = [0, 0, 0]

    with pytest.raises(ValueError, match="point_xy and color_hint"):
        _registry().validate("solve_pickup", grounding)


def test_contract_rejects_wrong_roles_and_out_of_bounds_points() -> None:
    with pytest.raises(ValueError, match="roles mismatch"):
        _registry().validate("solve_push_to_target", _point("object", color_hint="red"))

    with pytest.raises(ValueError, match="outside image bounds"):
        _registry().validate(
            "solve_pickup",
            {
                **_point("object", color_hint="red"),
                "targets": {
                    "object": {
                        "point_xy": [256, 10],
                        "color_hint": "red",
                    }
                },
            },
            image_shape=(256, 256),
        )


def test_contract_enforces_color_policy() -> None:
    with pytest.raises(ValueError, match="requires a color_hint"):
        _registry().validate("solve_pickup", _point("object"))

    with pytest.raises(ValueError, match="must not use color_hint"):
        _registry().validate(
            "solve_put_into_bin",
            _point("container", color_hint="white"),
        )


def test_peg_endpoint_color_is_limited_to_black_or_white() -> None:
    black = _registry().validate(
        "grasp_and_lift_peg_side",
        _point("peg", color_hint="BLACK"),
    )
    assert black["targets"]["peg"]["color_hint"] == "black"

    with pytest.raises(ValueError, match="must be one of"):
        _registry().validate(
            "grasp_and_lift_peg_side",
            _point("peg", color_hint="orange"),
        )


def test_whole_peg_grasp_uses_no_endpoint_color() -> None:
    grounding = _registry().validate(
        "grasp_and_lift_peg",
        _point("peg"),
    )
    assert grounding["targets"]["peg"]["color_hint"] is None

    with pytest.raises(ValueError, match="must not use color_hint"):
        _registry().validate(
            "grasp_and_lift_peg",
            _point("peg", color_hint="black"),
        )


def test_peg_push_target_uses_trusted_fixed_color_not_vlm_color() -> None:
    grounding = {
        "status": "ok",
        "image_ref": "current_front",
        "targets": {
            "object": {"point_xy": [80, 90], "color_hint": "red"},
            "target": {"point_xy": [160, 120], "color_hint": None},
        },
    }

    assert _registry().validate(
        "solve_push_to_target_with_peg",
        grounding,
        image_shape=(256, 256),
    ) == grounding

    grounding["targets"]["target"]["color_hint"] = "purple"
    with pytest.raises(ValueError, match="trusted fixed_color"):
        _registry().validate("solve_push_to_target_with_peg", grounding)


@pytest.mark.parametrize(
    "action_name",
    [
        "solve_putdown",
        "solve_hold_abs_timestep",
        "solve_pattern_direction",
        "solve_swingonto_with_direction",
    ],
)
def test_actions_without_vlm_points_require_not_required_status(
    action_name: str,
) -> None:
    assert _registry().validate(action_name, _not_required()) == _not_required()

    with pytest.raises(ValueError, match="must not receive VLM target points"):
        _registry().validate(action_name, _point("target"))
