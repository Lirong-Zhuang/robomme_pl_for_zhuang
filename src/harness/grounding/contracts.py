"""Validate VLM 2D hints against trusted action perception contracts.

The VLM is allowed to identify a target only by one image-space point and an
optional color hint.  OpenCV owns the mask, and F2G owns all metric geometry.
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


def _required_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


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

    def contract(self, action_name: str) -> dict[str, Any]:
        """Return a defensive copy of one trusted perception contract."""

        try:
            contract = self._contracts[action_name]
        except KeyError as error:
            raise ValueError(
                f"unknown action perception contract: {action_name!r}"
            ) from error
        return copy.deepcopy(contract)
