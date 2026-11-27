"""Validate VLM 2D hints against trusted action perception contracts.

The VLM is allowed to identify a target only by one image-space point and an
optional color hint.  The trusted catalog, never the VLM, selects the geometry
backend for each target role.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_ROLE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_GROUNDING_STATUSES = {"ok", "ambiguous", "not_required"}
_GROUNDING_FIELDS = {"status", "image_ref", "targets"}
_GROUNDING_TARGET_FIELDS = {"point_xy", "color_hint"}
_VLM_POINT_BACKENDS = {
    "grasp_grounding",
    "side_grasp_grounding",
    "location_grounding",
}
VLM_POINT_COORDINATE_SPACES = {"pixel", "normalized_1000"}
_GROUNDING_STATUS_ALIASES = {
    "visible": "ok",
    "found": "ok",
    "unclear": "ambiguous",
    "not_visible": "ambiguous",
    "none": "not_required",
    "not_needed": "not_required",
}
_CURRENT_IMAGE_REF_ALIASES = {
    "current_overall": "current_front",
    "current_front_overall": "current_front",
}
_ROLE_ALIASES = {
    "object": {"cube", "obj"},
    "container": {"bin"},
    "button": {"button_cap"},
    "target": {"goal", "destination"},
    "peg": {"rod"},
    "box": {"insertion_box"},
}


def _required_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def pack_grasp_repair_grounding(contract: Mapping[str, Any], reply: Any) -> dict[str, Any]:
    """Bind a flat grasp reply to its trusted role and current image."""
    if contract.get("mode") != "vlm_points" or len(contract["targets"]) != 1:
        raise ValueError("flat grasp repair requires exactly one VLM target role")
    if not isinstance(reply, Mapping) or set(reply) != _GROUNDING_TARGET_FIELDS:
        raise ValueError("grasp repair must contain exactly point_xy and color_hint")
    if reply["point_xy"] is None:
        if reply["color_hint"] is not None:
            raise ValueError("an unclear grasp requires point_xy=null and color_hint=null")
        return {"status": "ambiguous", "image_ref": "current_front", "targets": {}}

    role = contract["targets"][0]["role"]
    return {"status": "ok", "image_ref": "current_front", "targets": {role: dict(reply)}}


def normalize_vlm_grounding(value: Any) -> dict[str, Any]:
    """Normalize the complete VLM-owned payload and reject any 3D fields."""

    if not isinstance(value, Mapping) or set(value) != _GROUNDING_FIELDS:
        raise ValueError(
            "grounding must contain exactly status, image_ref, and targets"
        )
    status = _required_string("grounding.status", value.get("status"))
    if status not in _GROUNDING_STATUSES:
        raise ValueError(f"invalid grounding status: {status!r}")
    image_ref = value.get("image_ref")
    targets = value.get("targets")
    if not isinstance(targets, Mapping):
        raise ValueError("grounding.targets must be an object")

    if status == "not_required":
        if image_ref is not None or targets:
            raise ValueError(
                "not_required grounding requires image_ref=null and targets={}"
            )
        return {"status": status, "image_ref": None, "targets": {}}

    image_ref = _required_string("grounding.image_ref", image_ref)
    if status == "ambiguous":
        if targets:
            raise ValueError("ambiguous grounding requires an empty targets object")
        return {"status": status, "image_ref": image_ref, "targets": {}}

    if not targets:
        raise ValueError("ok grounding requires at least one target point")
    normalized_targets: dict[str, Any] = {}
    for raw_role, raw_target in targets.items():
        role = _required_string("grounding target role", raw_role)
        if not _ROLE_PATTERN.fullmatch(role):
            raise ValueError(f"invalid grounding target role: {role!r}")
        if (
            not isinstance(raw_target, Mapping)
            or set(raw_target) != _GROUNDING_TARGET_FIELDS
        ):
            raise ValueError(
                f"grounding target {role!r} must contain exactly "
                "point_xy and color_hint"
            )
        point = raw_target.get("point_xy")
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(
                f"grounding target {role!r} point_xy must contain x and y"
            )
        if any(
            isinstance(component, bool)
            or not isinstance(component, (int, float))
            for component in point
        ):
            raise ValueError(f"grounding target {role!r} point_xy must be numeric")
        point_xy = [float(component) for component in point]
        if any(
            not math.isfinite(component) or component < 0
            for component in point_xy
        ):
            raise ValueError(
                f"grounding target {role!r} point_xy must be finite and non-negative"
            )
        color_hint = raw_target.get("color_hint")
        if color_hint is not None:
            color_hint = _required_string(
                f"grounding target {role!r} color_hint", color_hint
            )
        normalized_targets[role] = {
            "point_xy": point_xy,
            "color_hint": color_hint,
        }
    return {
        "status": status,
        "image_ref": image_ref,
        "targets": normalized_targets,
    }


def convert_vlm_grounding_to_pixels(
    value: Any,
    *,
    image_shape: tuple[int, int],
    coordinate_space: str,
) -> dict[str, Any]:
    """Convert one declared VLM coordinate space into trusted image pixels."""

    if coordinate_space not in VLM_POINT_COORDINATE_SPACES:
        raise ValueError(
            "coordinate_space must be one of "
            f"{sorted(VLM_POINT_COORDINATE_SPACES)}, got {coordinate_space!r}"
        )
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("image_shape dimensions must be positive")

    normalized = normalize_vlm_grounding(value)
    if normalized["status"] != "ok" or coordinate_space == "pixel":
        return normalized

    for role, target in normalized["targets"].items():
        x, y = target["point_xy"]
        if x > 1000 or y > 1000:
            raise ValueError(
                f"grounding role {role!r} normalized point_xy must be within "
                "the inclusive 0-1000 range"
            )
        target["point_xy"] = [
            min(float(width - 1), x * width / 1000),
            min(float(height - 1), y * height / 1000),
        ]
    return normalized


class GroundingContractRegistry:
    """Action-indexed validator for the trusted catalog perception section."""

    def __init__(self, contracts: Mapping[str, Mapping[str, Any]]) -> None:
        self._contracts = {
            _required_string("action name", name): dict(contract)
            for name, contract in contracts.items()
        }

    @classmethod
    def from_catalog(cls, path: Path) -> GroundingContractRegistry:
        catalog = json.loads(Path(path).read_text(encoding="utf-8"))
        if catalog.get("schema_version") != "action_catalog.v2":
            raise ValueError("grounding requires action_catalog.v2")
        actions = catalog.get("actions")
        if not isinstance(actions, list):
            raise ValueError("action catalog must contain an actions list")
        return cls(
            {
                action["name"]: action["perception"]
                for action in actions
                if isinstance(action, Mapping)
            }
        )

    def validate(
        self,
        action_name: str,
        grounding: Any,
        *,
        image_shape: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Validate roles, color policy, and optional image bounds."""

        try:
            contract = self._contracts[action_name]
        except KeyError as error:
            raise ValueError(f"unknown action perception contract: {action_name!r}") from error
        mode = contract.get("mode")
        targets = contract.get("targets")
        if mode not in {"none", "vlm_points", "scene_target_set"}:
            raise ValueError(f"invalid perception mode for {action_name!r}: {mode!r}")
        if not isinstance(targets, list):
            raise ValueError(f"perception targets for {action_name!r} must be a list")

        for target in targets:
            if not isinstance(target, Mapping):
                raise ValueError(
                    f"perception target for {action_name!r} must be an object"
                )
            role = _required_string("perception target role", target.get("role"))
            point_semantics = _required_string(
                f"perception target {role!r} point_semantics",
                target.get("point_semantics"),
            )
            backend = target.get("grounding_backend")
            if mode == "vlm_points":
                normalized_semantics = point_semantics.casefold()
                if not any(
                    word in normalized_semantics for word in ("center", "centroid")
                ):
                    raise ValueError(
                        f"grounding role {role!r} point_semantics must request "
                        "a center or centroid"
                    )
                if backend not in _VLM_POINT_BACKENDS:
                    raise ValueError(
                        f"grounding role {role!r} has invalid grounding_backend: "
                        f"{backend!r}"
                    )
                if backend == "location_grounding":
                    if target.get("geometry_outputs") != ["position"]:
                        raise ValueError(
                            f"location grounding role {role!r} must output position"
                        )
                    if "segmentation_profile" in target or "f2g_outputs" in target:
                        raise ValueError(
                            f"location grounding role {role!r} must not configure "
                            "OpenCV or F2G"
                        )
                else:
                    _required_string(
                        f"grounding role {role!r} segmentation_profile",
                        target.get("segmentation_profile"),
                    )
                    f2g_outputs = target.get("f2g_outputs")
                    if not isinstance(f2g_outputs, list) or "pose" not in f2g_outputs:
                        raise ValueError(
                            f"grounding role {role!r} must request an F2G pose"
                        )
                if (
                    backend == "side_grasp_grounding"
                    and target.get("color_hint") != "required"
                ):
                    raise ValueError(
                        f"side grounding role {role!r} requires a color hint"
                    )
            elif mode == "scene_target_set" and backend != "scene_grounding":
                raise ValueError(
                    f"scene target role {role!r} requires scene_grounding"
                )

        normalized = normalize_vlm_grounding(grounding)
        if mode != "vlm_points":
            if normalized["status"] != "not_required":
                raise ValueError(
                    f"action {action_name!r} must not receive VLM target points"
                )
            return normalized
        if normalized["status"] == "ambiguous":
            return normalized
        if normalized["status"] != "ok":
            raise ValueError(f"action {action_name!r} requires VLM target points")

        expected = {target["role"]: target for target in targets}
        actual = normalized["targets"]
        if set(actual) != set(expected):
            raise ValueError(
                f"action {action_name!r} grounding roles mismatch; "
                f"expected={sorted(expected)}, actual={sorted(actual)}"
            )
        for role, point_hint in actual.items():
            color_policy = expected[role]["color_hint"]
            color_hint = point_hint["color_hint"]
            if color_policy == "required" and color_hint is None:
                raise ValueError(f"grounding role {role!r} requires a color_hint")
            allowed_colors = expected[role].get("allowed_colors")
            if allowed_colors is not None:
                if (
                    not isinstance(allowed_colors, list)
                    or not allowed_colors
                    or any(
                        not isinstance(color, str) or not color.strip()
                        for color in allowed_colors
                    )
                ):
                    raise ValueError(
                        f"grounding role {role!r} has invalid allowed_colors"
                    )
                normalized_allowed = {
                    color.strip().lower() for color in allowed_colors
                }
                normalized_color = (
                    color_hint.strip().lower()
                    if isinstance(color_hint, str)
                    else None
                )
                if normalized_color not in normalized_allowed:
                    raise ValueError(
                        f"grounding role {role!r} color_hint must be one of "
                        f"{sorted(normalized_allowed)}"
                    )
                point_hint["color_hint"] = normalized_color
            if color_policy == "not_used" and color_hint is not None:
                raise ValueError(f"grounding role {role!r} must not use color_hint")
            if color_policy == "fixed":
                fixed_color = expected[role].get("fixed_color")
                if not isinstance(fixed_color, str) or not fixed_color.strip():
                    raise ValueError(
                        f"grounding role {role!r} has no trusted fixed_color"
                    )
                if color_hint is not None:
                    raise ValueError(
                        f"grounding role {role!r} uses trusted fixed_color and "
                        "must not receive a VLM color_hint"
                    )
            if image_shape is not None:
                height, width = image_shape
                x, y = point_hint["point_xy"]
                if height <= 0 or width <= 0:
                    raise ValueError("image_shape dimensions must be positive")
                if not (0 <= x < width and 0 <= y < height):
                    raise ValueError(
                        f"grounding role {role!r} point_xy is outside image bounds"
                    )
        return normalized

    def adapt_vlm_output(
        self,
        action_name: str,
        grounding: Any,
        *,
        semantic_arguments: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Canonicalize harmless model-format drift from a trusted contract.

        Perception mode, a sole target's role name, and non-VLM color policies
        are already determined by the trusted action. Required colors,
        coordinates, image references, and multi-role identity remain
        model-owned and are still validated strictly.
        """

        contract = self.contract(action_name)
        if contract.get("mode") != "vlm_points":
            return {
                "status": "not_required",
                "image_ref": None,
                "targets": {},
            }

        prepared = copy.deepcopy(grounding)
        if isinstance(prepared, dict):
            raw_status = prepared.get("status")
            if isinstance(raw_status, str):
                status = raw_status.strip().lower().replace("-", "_").replace(
                    " ", "_"
                )
                prepared["status"] = _GROUNDING_STATUS_ALIASES.get(
                    status,
                    status,
                )
            raw_image_ref = prepared.get("image_ref")
            if isinstance(raw_image_ref, str):
                prepared["image_ref"] = _CURRENT_IMAGE_REF_ALIASES.get(
                    raw_image_ref.strip().lower(),
                    raw_image_ref.strip(),
                )

        if isinstance(prepared, Mapping) and prepared.get("status") == "ok":
            raw_targets = prepared.get("targets")
            if isinstance(raw_targets, Mapping):
                for raw_target in raw_targets.values():
                    if not isinstance(raw_target, dict):
                        continue
                    if (
                        "point_xy" not in raw_target
                        and {"x", "y"} <= set(raw_target)
                        and set(raw_target)
                        <= {"x", "y", "color", "color_hint"}
                    ):
                        raw_target["point_xy"] = [
                            raw_target.pop("x"),
                            raw_target.pop("y"),
                        ]
                    if "color" in raw_target and "color_hint" not in raw_target:
                        raw_target["color_hint"] = raw_target.pop("color")
                    if set(raw_target) == {"point_xy"}:
                        # An omitted color_hint is equivalent to JSON null;
                        # policy validation below still rejects null when a
                        # genuinely model-selected color is required.
                        raw_target["color_hint"] = None

        normalized = normalize_vlm_grounding(prepared)
        if normalized["status"] != "ok":
            return normalized

        target_specs = contract["targets"]
        targets = normalized["targets"]
        expected_roles = {item["role"] for item in target_specs}
        actual_roles = set(targets)
        if expected_roles <= actual_roles:
            # Some VLMs carry the previous action's target into the next
            # response (for example object + container for put_into_bin).
            # The trusted action contract owns the required role set, so these
            # extra role entries are irrelevant and safe to discard.
            targets = {
                role: target
                for role, target in targets.items()
                if role in expected_roles
            }
            normalized["targets"] = targets
        elif len(target_specs) == 1 and len(targets) == 1:
            expected_role = target_specs[0]["role"]
            actual_role = next(iter(targets))
            if actual_role in _ROLE_ALIASES.get(expected_role, set()):
                targets = {expected_role: targets[actual_role]}
                normalized["targets"] = targets

        specs_by_role = {item["role"]: item for item in target_specs}
        semantic_color = None
        if isinstance(semantic_arguments, Mapping):
            candidate = semantic_arguments.get("color")
            if isinstance(candidate, str) and candidate.strip():
                semantic_color = candidate.strip().lower()
        for role, target in targets.items():
            spec = specs_by_role.get(role)
            if spec is not None and spec.get("color_hint") in {
                "not_used",
                "fixed",
            }:
                target["color_hint"] = None
            elif (
                spec is not None
                and spec.get("color_hint") == "required"
                and semantic_color is not None
            ):
                grounded_color = target.get("color_hint")
                if grounded_color is None:
                    target["color_hint"] = semantic_color
                elif grounded_color.strip().lower() != semantic_color:
                    raise ValueError(
                        f"grounding role {role!r} color_hint conflicts with "
                        f"subgoal color {semantic_color!r}"
                    )
        return normalized

    def contract(self, action_name: str) -> dict[str, Any]:
        """Return a defensive copy of one trusted perception contract."""

        try:
            contract = self._contracts[action_name]
        except KeyError as error:
            raise ValueError(
                f"unknown action perception contract: {action_name!r}"
            ) from error
        return copy.deepcopy(contract)
