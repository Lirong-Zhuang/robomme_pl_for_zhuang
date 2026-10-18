"""Deterministic gripper gate for Reporter ``pick up`` predictions.

RoboMME exposes each observed finger opening in metres in the range
``[0.0, 0.04]``.  A failed empty grasp can close to the lower endpoint, while
an open gripper remains near the upper endpoint.  For a ``pick up`` subgoal,
this tool accepts a Reporter ``true`` only when the measured opening is outside
configurable endpoint buffers around both fully closed and fully open.

The tool is intentionally only a veto: it never turns a Reporter ``false``
into ``true`` and it is not applicable to subgoals without the literal phrase
``pick up`` (case-insensitive).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


DEFAULT_GRIPPER_CLOSED_POSITION = 0.0
DEFAULT_GRIPPER_OPEN_POSITION = 0.04
DEFAULT_GRIPPER_ENDPOINT_BUFFER = 0.002


@dataclass(frozen=True)
class GripperGateResult:
    """Explain one attempted validation of a Reporter ``true`` result."""

    applicable: bool
    passed: bool
    reason: str
    gripper_position: float
    accepted_lower_bound: float
    accepted_upper_bound: float


class PickUpGripperGate:
    """Veto ``pick up`` completion when the gripper is at either endpoint."""

    def __init__(
        self,
        *,
        closed_position: float = DEFAULT_GRIPPER_CLOSED_POSITION,
        open_position: float = DEFAULT_GRIPPER_OPEN_POSITION,
        endpoint_buffer: float = DEFAULT_GRIPPER_ENDPOINT_BUFFER,
    ) -> None:
        values = (closed_position, open_position, endpoint_buffer)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Gripper gate positions and buffer must be finite")
        if open_position <= closed_position:
            raise ValueError(
                "Gripper open position must be greater than closed position"
            )
        if endpoint_buffer < 0:
            raise ValueError("Gripper endpoint buffer must be non-negative")
        if 2 * endpoint_buffer >= open_position - closed_position:
            raise ValueError(
                "Gripper endpoint buffers leave no intermediate acceptance range"
            )

        self.closed_position = float(closed_position)
        self.open_position = float(open_position)
        self.endpoint_buffer = float(endpoint_buffer)

    @property
    def accepted_lower_bound(self) -> float:
        return self.closed_position + self.endpoint_buffer

    @property
    def accepted_upper_bound(self) -> float:
        return self.open_position - self.endpoint_buffer

    @staticmethod
    def is_pick_up_subgoal(subgoal: str | None) -> bool:
        """Return whether the subgoal contains the literal phrase ``pick up``."""
        return subgoal is not None and "pick up" in subgoal.casefold()

    def verify_reporter_true(
        self,
        *,
        subgoal: str | None,
        gripper_position: float,
    ) -> GripperGateResult:
        """Check one Reporter ``true`` candidate.

        Callers must invoke this method only for a parsed Reporter ``true``.
        Non-``pick up`` subgoals pass through as not applicable.  For a
        ``pick up`` subgoal, values inside either endpoint buffer are rejected;
        only the strict intermediate interval is accepted.
        """
        position = float(gripper_position)
        lower = self.accepted_lower_bound
        upper = self.accepted_upper_bound

        if not self.is_pick_up_subgoal(subgoal):
            return GripperGateResult(
                applicable=False,
                passed=True,
                reason="subgoal_does_not_contain_pick_up",
                gripper_position=position,
                accepted_lower_bound=lower,
                accepted_upper_bound=upper,
            )
        if not math.isfinite(position):
            return GripperGateResult(
                applicable=True,
                passed=False,
                reason="non_finite_gripper_position",
                gripper_position=position,
                accepted_lower_bound=lower,
                accepted_upper_bound=upper,
            )
        if position <= lower:
            return GripperGateResult(
                applicable=True,
                passed=False,
                reason="fully_closed_or_inside_closed_buffer",
                gripper_position=position,
                accepted_lower_bound=lower,
                accepted_upper_bound=upper,
            )
        if position >= upper:
            return GripperGateResult(
                applicable=True,
                passed=False,
                reason="fully_open_or_inside_open_buffer",
                gripper_position=position,
                accepted_lower_bound=lower,
                accepted_upper_bound=upper,
            )
        return GripperGateResult(
            applicable=True,
            passed=True,
            reason="intermediate_gripper_position",
            gripper_position=position,
            accepted_lower_bound=lower,
            accepted_upper_bound=upper,
        )
