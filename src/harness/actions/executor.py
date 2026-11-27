"""Dispatch explicit-geometry functions through Oracle Plus runtime execution."""

from __future__ import annotations

from typing import Any
from typing import Mapping

from .env_targets import bind_environment_geometry
from .registry import CatalogAction
from .types import ActionRequest
from .types import ActionResult


class ActionExecutor:
    """Bind grounded geometry, then call the function inside runtime.execute()."""

    def __init__(self, runtime: Any, registry: Mapping[str, CatalogAction]) -> None:
        if getattr(runtime, "env", None) is None:
            raise ValueError("Oracle Plus runtime must own an environment")
        if getattr(runtime, "planner", None) is None:
            raise ValueError("Oracle Plus runtime must be reset before execution")
        self.runtime = runtime
        self.registry = dict(registry)
        self.last_execution: Any | None = None

    def execute(
        self,
        request: ActionRequest,
        *,
        step_callback: Any | None = None,
    ) -> ActionResult:
        try:
            action = self.registry[request.action_name]
        except KeyError as error:
            raise ValueError(f"unregistered action: {request.action_name!r}") from error

        # Bind before entering the collector so malformed perception data cannot
        # partially move the robot.
        request = bind_environment_geometry(self.runtime.env, request)
        kwargs = action.bind(request)

        def solve() -> Any:
            return action.solve(
                self.runtime.env,
                self.runtime.planner,
                **kwargs,
            )

        if step_callback is None:
            execution = self.runtime.execute(solve)
        else:
            execution = self.runtime.execute(solve, step_callback=step_callback)
        self.last_execution = execution
        return ActionResult(
            status=execution.status,
            stopped=execution.stopped,
            simulator_frame_count=execution.simulator_frame_count,
            info=execution.info,
        )
