"""Trusted geometry-driven actions for the Agentic harness."""

from typing import Any

from .types import ActionRequest
from .types import ActionResult
from .types import ObjectEstimate
from .types import Pose3D
from .types import Size3D


__all__ = [
    "ActionExecutor",
    "ActionRequest",
    "ActionResult",
    "CatalogAction",
    "ObjectEstimate",
    "Pose3D",
    "Size3D",
    "build_action_registry",
    "resolve_action_arguments",
]


def __getattr__(name: str) -> Any:
    """Avoid loading SAPIEN/MPLib when only geometry types are needed."""

    if name == "ActionExecutor":
        from .executor import ActionExecutor

        return ActionExecutor
    if name in {"CatalogAction", "build_action_registry"}:
        from . import registry

        return getattr(registry, name)
    if name == "resolve_action_arguments":
        from .semantic_arguments import resolve_action_arguments

        return resolve_action_arguments
    raise AttributeError(name)
