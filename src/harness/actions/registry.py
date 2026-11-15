"""Bind perception geometry to Oracle-style function parameters."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .button import solve_button
from .button import solve_button_ready
from .oracle_reference import ORACLE_ACTION_REFERENCES
from .peg import grasp_and_lift_peg_side
from .peg import insert_peg
from .pick_place import solve_pickup
from .pick_place import solve_pickup_bin
from .pick_place import solve_put_into_bin
from .pick_place import solve_putdown
from .pick_place import solve_putonto_target
from .pick_place import solve_swingonto_whenhold
from .push import solve_push_to_target
from .push import solve_push_to_target_with_peg
from .temporal import solve_hold_abs_timestep
from .trajectory import solve_pattern_direction
from .trajectory import solve_swingonto_with_direction
from .types import ActionRequest


@dataclass(frozen=True)
class TargetBinding:
    """Map one grounded target role to explicit Oracle-function kwargs."""

    role: str
    pose_argument: str
    size_argument: str | None = None
    label_argument: str | None = None


@dataclass(frozen=True)
class TargetSetBinding:
    """Map a perceived candidate set to a tuple of explicit poses."""

    role: str
    poses_argument: str


@dataclass(frozen=True)
class CatalogAction:
    """Trusted function plus deterministic perception-argument binding."""

    name: str
    solve: Callable[..., Any]
    targets: tuple[TargetBinding, ...] = ()
    target_sets: tuple[TargetSetBinding, ...] = ()

    def _validate_pose_frames(self, request: ActionRequest) -> None:
        estimates = list(request.targets.values())
        for target_set in request.target_sets.values():
            estimates.extend(target_set)
        invalid = {
            estimate.pose.frame
            for estimate in estimates
            if estimate.pose.frame not in {"world", "base"}
        }
        if invalid:
            raise ValueError(
                "targets must be converted to world/base frame before execution; "
                f"received {sorted(invalid)}"
            )

    def bind(self, request: ActionRequest) -> dict[str, Any]:
        if request.action_name != self.name:
            raise ValueError(
                f"request action {request.action_name!r} does not match {self.name!r}"
            )
        self._validate_pose_frames(request)
        signature = inspect.signature(self.solve)
        accepted_arguments = set(signature.parameters).difference({"env", "planner"})
        kwargs = {
            key: value
            for key, value in request.arguments.items()
            if key in accepted_arguments
        }
        for binding in self.targets:
            try:
                estimate = request.targets[binding.role]
            except KeyError as error:
                raise ValueError(
                    f"action {self.name!r} is missing target role {binding.role!r}"
                ) from error
            if binding.pose_argument in request.arguments:
                raise ValueError(
                    f"argument {binding.pose_argument!r} is owned by grounding"
                )
            kwargs[binding.pose_argument] = estimate.pose
            if binding.size_argument is not None:
                if binding.size_argument in request.arguments:
                    raise ValueError(
                        f"argument {binding.size_argument!r} is owned by grounding"
                    )
                kwargs[binding.size_argument] = estimate.size
            if binding.label_argument is not None:
                if binding.label_argument in request.arguments:
                    raise ValueError(
                        f"argument {binding.label_argument!r} is owned by grounding"
                    )
                if not isinstance(estimate.label, str) or not estimate.label.strip():
                    raise ValueError(
                        f"action {self.name!r} target role {binding.role!r} "
                        "requires a grounded label"
                    )
                kwargs[binding.label_argument] = estimate.label.strip().lower()
        for binding in self.target_sets:
            try:
                estimates = request.target_sets[binding.role]
            except KeyError as error:
                raise ValueError(
                    f"action {self.name!r} is missing target set {binding.role!r}"
                ) from error
            if not estimates:
                raise ValueError(
                    f"action {self.name!r} target set {binding.role!r} is empty"
                )
            if binding.poses_argument in request.arguments:
                raise ValueError(
                    f"argument {binding.poses_argument!r} is owned by grounding"
                )
            kwargs[binding.poses_argument] = tuple(item.pose for item in estimates)
        signature.bind(None, None, **kwargs)
        return kwargs

    def __call__(self, env: Any, planner: Any, request: ActionRequest) -> Any:
        """Call the Oracle-style function with deterministically bound geometry."""

        return self.solve(env, planner, **self.bind(request))


def build_action_registry() -> dict[str, CatalogAction]:
    """Build all catalog bindings without changing the action implementations."""

    actions = (
        CatalogAction(
            "solve_pickup",
            solve_pickup,
            targets=(TargetBinding("object", "obj_pose", "obj_size"),),
        ),
        CatalogAction(
            "solve_put_into_bin",
            solve_put_into_bin,
            targets=(TargetBinding("container", "target_pose"),),
        ),
        CatalogAction(
            "solve_button",
            solve_button,
            targets=(TargetBinding("button", "button_pose"),),
        ),
        CatalogAction(
            "grasp_and_lift_peg_side",
            grasp_and_lift_peg_side,
            targets=(TargetBinding("peg", "peg_pose", label_argument="peg_color"),),
        ),
        CatalogAction(
            "solve_push_to_target_with_peg",
            solve_push_to_target_with_peg,
            targets=(
                TargetBinding("object", "obj_pose"),
                TargetBinding("target", "target_pose"),
            ),
        ),
        CatalogAction(
            "solve_push_to_target",
            solve_push_to_target,
            targets=(
                TargetBinding("object", "obj_pose", "obj_size"),
                TargetBinding("target", "target_pose"),
            ),
        ),
        CatalogAction(
            "solve_putonto_target",
            solve_putonto_target,
            targets=(TargetBinding("target", "target_pose"),),
        ),
        CatalogAction("solve_putdown", solve_putdown),
        CatalogAction(
            "solve_button_ready",
            solve_button_ready,
            targets=(TargetBinding("button", "button_pose"),),
        ),
        CatalogAction("solve_hold_abs_timestep", solve_hold_abs_timestep),
        CatalogAction(
            "solve_pickup_bin",
            solve_pickup_bin,
            targets=(TargetBinding("object", "obj_pose", "obj_size"),),
        ),
        CatalogAction(
            "solve_swingonto_whenhold",
            solve_swingonto_whenhold,
            targets=(TargetBinding("target", "target_pose"),),
        ),
        CatalogAction(
            "insert_peg",
            insert_peg,
            targets=(
                TargetBinding("peg", "peg_pose"),
                TargetBinding("box", "box_pose"),
            ),
        ),
        CatalogAction(
            "solve_pattern_direction",
            solve_pattern_direction,
            target_sets=(TargetSetBinding("targets", "target_poses"),),
        ),
        CatalogAction(
            "solve_swingonto_with_direction",
            solve_swingonto_with_direction,
            target_sets=(TargetSetBinding("targets", "target_poses"),),
        ),
    )
    registry = {action.name: action for action in actions}
    if len(registry) != len(actions):
        raise RuntimeError("duplicate action name in registry")
    if set(registry) != set(ORACLE_ACTION_REFERENCES):
        raise RuntimeError("action registry and Oracle reference map do not match")
    return registry
