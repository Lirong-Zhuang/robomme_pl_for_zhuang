"""Resolve simulator-owned targets used by privileged Oracle-style actions."""

from __future__ import annotations

from typing import Any

import numpy as np


def _base_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def actor_position(actor: Any, *, target_name: str) -> list[float]:
    """Return the first environment actor position as three finite floats."""

    pose = getattr(actor, "pose", None)
    position = getattr(pose, "p", None)
    if position is None:
        raise RuntimeError(f"environment target {target_name!r} has no pose.p")
    if hasattr(position, "detach"):
        position = position.detach()
    if hasattr(position, "cpu"):
        position = position.cpu()
    if hasattr(position, "numpy"):
        position = position.numpy()
    values = np.asarray(position, dtype=np.float64).reshape(-1, 3)[0]
    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            f"environment target {target_name!r} has a non-finite position"
        )
    return [float(item) for item in values]


def bin_position(env: Any) -> list[float]:
    """Match Oracle Plus by reading BinFill's board-with-hole actor directly."""

    base = _base_env(env)
    actor = getattr(base, "board_with_hole", None)
    if actor is None:
        raise RuntimeError("environment bin target 'board_with_hole' is unavailable")
    return actor_position(actor, target_name="board_with_hole")


def button_position(env: Any, *, ordinal: str | None = None) -> list[float]:
    """Read the task's semantic button actor directly from the environment."""

    base = _base_env(env)
    task_name = type(base).__name__.casefold()
    normalized_ordinal = ordinal.strip().casefold() if ordinal else None
    if normalized_ordinal == "first":
        names = ("button_left", "button")
    elif normalized_ordinal == "second":
        names = ("button_right", "button")
    elif normalized_ordinal is not None:
        raise ValueError(f"unsupported button ordinal: {ordinal!r}")
    else:
        # Oracle Plus uses button_left for ButtonUnmask variants and button for
        # the other single-button tasks. Keep the same preference and fallback.
        names = (
            ("button_left", "button")
            if "buttonunmask" in task_name
            else ("button", "button_left")
        )
    for name in names:
        actor = getattr(base, name, None)
        if actor is not None:
            return actor_position(actor, target_name=name)
    raise RuntimeError(
        f"environment button target is unavailable; tried {list(names)}"
    )
