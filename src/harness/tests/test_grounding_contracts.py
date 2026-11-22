from __future__ import annotations

from pathlib import Path

import pytest

from harness.grounding import GroundingContractRegistry
from harness.grounding import convert_vlm_grounding_to_pixels


CATALOG_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "action_candidates"
    / "action_catalog.json"
)


def _registry() -> GroundingContractRegistry:
    return GroundingContractRegistry.from_catalog(CATALOG_PATH)


def _point(
    role: str,
    *,
    point_xy: list[float] | None = None,
    color_hint: str | None = None,
) -> dict:
    return {
        "status": "ok",
        "image_ref": "current_front",
        "targets": {
            role: {
                "point_xy": point_xy or [120, 90],
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


def test_normalized_1000_coordinates_are_converted_before_validation() -> None:
    grounding = {
        **_point("object", color_hint="red"),
        "targets": {
            "object": {
                "point_xy": [500, 250],
                "color_hint": "red",
            }
        },
    }

    converted = convert_vlm_grounding_to_pixels(
        grounding,
        image_shape=(256, 256),
        coordinate_space="normalized_1000",
    )

    assert converted["targets"]["object"]["point_xy"] == pytest.approx(
        [128.0, 64.0]
    )
    assert _registry().validate(
        "solve_pickup",
        converted,
        image_shape=(256, 256),
    )["targets"]["object"]["point_xy"] == pytest.approx([128.0, 64.0])


def test_normalized_conversion_is_not_guessed_from_point_magnitude() -> None:
    grounding = {
        **_point("object", color_hint="red"),
        "targets": {
            "object": {
                "point_xy": [100, 100],
                "color_hint": "red",
            }
        },
    }

    normalized = convert_vlm_grounding_to_pixels(
        grounding,
        image_shape=(256, 256),
        coordinate_space="normalized_1000",
    )
    pixels = convert_vlm_grounding_to_pixels(
        grounding,
        image_shape=(256, 256),
        coordinate_space="pixel",
    )

    assert normalized["targets"]["object"]["point_xy"] == pytest.approx(
        [25.6, 25.6]
    )
    assert pixels["targets"]["object"]["point_xy"] == [100.0, 100.0]


def test_normalized_coordinates_above_1000_are_rejected() -> None:
    grounding = {
        **_point("object", color_hint="red"),
        "targets": {
            "object": {
                "point_xy": [1001, 500],
                "color_hint": "red",
            }
        },
    }

    with pytest.raises(ValueError, match="inclusive 0-1000"):
        convert_vlm_grounding_to_pixels(
            grounding,
            image_shape=(256, 256),
            coordinate_space="normalized_1000",
        )


def test_normalized_1000_boundary_is_clamped_inside_image() -> None:
    grounding = {
        **_point("object", color_hint="red"),
        "targets": {
            "object": {
                "point_xy": [1000, 1000],
                "color_hint": "red",
            }
        },
    }

    converted = convert_vlm_grounding_to_pixels(
        grounding,
        image_shape=(256, 256),
        coordinate_space="normalized_1000",
    )

    assert converted["targets"]["object"]["point_xy"] == [255.0, 255.0]


def test_adapter_uses_trusted_single_role_and_ignores_forbidden_color() -> None:
    adapted = _registry().adapt_vlm_output(
        "solve_put_into_bin",
        _point("bin", point_xy=[500, 500], color_hint="gray"),
    )

    assert adapted == _point(
        "container",
        point_xy=[500.0, 500.0],
        color_hint=None,
    )


def test_adapter_discards_points_for_action_with_no_vlm_perception() -> None:
    adapted = _registry().adapt_vlm_output(
        "solve_putdown",
        _point("table", point_xy=[500, 500], color_hint="brown"),
    )

    assert adapted == {
        "status": "not_required",
        "image_ref": None,
        "targets": {},
    }


def test_adapter_fills_and_enforces_subgoal_color() -> None:
    without_color = _point("object", color_hint=None)
    without_color["targets"]["object"].pop("color_hint")
    adapted = _registry().adapt_vlm_output(
        "solve_pickup",
        without_color,
        semantic_arguments={"ordinal": "first", "color": "red"},
    )
    assert adapted["targets"]["object"]["color_hint"] == "red"

    with pytest.raises(ValueError, match="conflicts with subgoal color"):
        _registry().adapt_vlm_output(
            "solve_pickup",
            _point("object", color_hint="blue"),
            semantic_arguments={"ordinal": "first", "color": "red"},
        )


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
