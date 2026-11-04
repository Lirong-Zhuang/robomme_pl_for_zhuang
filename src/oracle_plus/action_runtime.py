"""Direct task-specific MPLib action execution for Oracle Plus."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable


class _DenseCollectionStopped(Exception):
    """Internal signal used to stop a planner after the env has stopped."""


def _flag_any(value: Any) -> bool:
    """Return whether a scalar/array/tensor termination flag contains true."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return bool(value.detach().bool().any().item())
    except ImportError:
        pass
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return bool(value.any())
    except ImportError:
        pass
    return bool(value)


def _run_with_bounded_dense_collection(
    planner: Any,
    solve: Callable[[], Any],
    *,
    max_retained_steps: int,
    planner_dense_step: Any,
    step_callback: Callable[[Any], None] | None = None,
) -> tuple[Any, int] | int:
    """Execute a planner action while retaining only its recent native steps.

    The benchmark's regular dense collector deep-copies every camera frame in a
    high-level motion.  A long path can therefore consume tens of gigabytes
    before the episode-level cleanup runs.  This Oracle Plus collector keeps a
    bounded tail for the VLM history and stops the remaining path as soon as the
    environment reports that the episode has ended.
    """
    retained: deque[Any] = deque(maxlen=max_retained_steps)
    total_step_count = 0
    original_step = planner.env.step

    def _step(action: Any) -> Any:
        nonlocal total_step_count
        out = original_step(action)
        steps = planner_dense_step._output_to_steps(out)
        total_step_count += len(steps)
        retained.extend(steps)
        if step_callback is not None:
            for step in steps:
                step_callback(step)
        if steps:
            _, _, terminated, truncated, info = steps[-1]
            status = info.get("status") if isinstance(info, dict) else None
            if (
                _flag_any(terminated)
                or _flag_any(truncated)
                or status in {"success", "fail", "timeout", "error"}
            ):
                raise _DenseCollectionStopped
        return out

    planner.env.step = _step
    try:
        try:
            result = solve()
        except _DenseCollectionStopped:
            result = None
        if result == -1:
            return -1
        return (
            planner_dense_step.to_step_batch(list(retained)),
            total_step_count,
        )
    finally:
        planner.env.step = original_step


@dataclass(frozen=True)
class ActionExecution:
    obs_batch: dict[str, list[Any]]
    info: dict[str, Any]
    status: str
    stopped: bool
    simulator_frame_count: int


class OraclePlusPlannerRuntime:
    """Own one RoboMME planner without using the multi_choice action space."""

    def __init__(
        self,
        env: Any,
        env_id: str,
        *,
        gui_render: bool = False,
        max_retained_steps: int = 3,
    ) -> None:
        if max_retained_steps < 1:
            raise ValueError("max_retained_steps must be at least 1")
        self.env = env
        self.env_id = env_id
        self.gui_render = gui_render
        self.max_retained_steps = max_retained_steps
        self.planner: Any = None
        self.front_camera_intrinsic: Any = None
        self.front_camera_extrinsic: Any = None
        self.front_image_shape: tuple[int, int] | None = None

    def _wrap_planner_retry(self, planner: Any, failure_type: type[Exception]) -> Any:
        original_screw = planner.move_to_pose_with_screw
        original_rrt = planner.move_to_pose_with_RRTStar

        def move_with_retry(*args: Any, **kwargs: Any) -> Any:
            for _ in range(3):
                try:
                    result = original_screw(*args, **kwargs)
                except failure_type:
                    continue
                if not (isinstance(result, int) and result == -1):
                    return result
            for _ in range(3):
                try:
                    result = original_rrt(*args, **kwargs)
                except Exception:
                    continue
                if not (isinstance(result, int) and result == -1):
                    return result
            raise RuntimeError("MPLib screw and RRT* planning retries were exhausted")

        planner.move_to_pose_with_screw = move_with_retry
        return planner

    def _build_planner(self) -> Any:
        try:
            from robomme.robomme_env.utils.planner_fail_safe import (
                FailAwarePandaArmMotionPlanningSolver,
                FailAwarePandaStickMotionPlanningSolver,
                ScrewPlanFailure,
            )
        except Exception:
            from mani_skill.examples.motionplanning.panda.motionplanner import (
                PandaArmMotionPlanningSolver as FailAwarePandaArmMotionPlanningSolver,
            )
            from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
                PandaStickMotionPlanningSolver as FailAwarePandaStickMotionPlanningSolver,
            )

            class ScrewPlanFailure(RuntimeError):
                pass

        planner_type = (
            FailAwarePandaStickMotionPlanningSolver
            if self.env_id in ("PatternLock", "RouteStick")
            else FailAwarePandaArmMotionPlanningSolver
        )
        kwargs: dict[str, Any] = {
            "debug": False,
            "vis": self.gui_render,
            "base_pose": self.env.unwrapped.agent.robot.pose,
            "visualize_target_grasp_pose": False,
            "print_env_info": False,
        }
        if self.env_id in ("PatternLock", "RouteStick"):
            kwargs["joint_vel_limits"] = 0.3
        planner = planner_type(self.env, **kwargs)
        return self._wrap_planner_retry(planner, ScrewPlanFailure)

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        self.planner = self._build_planner()
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        self._update_camera_cache(obs, info)
        return obs, info

    @staticmethod
    def _last_columnar(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return value[-1] if value else None
        return value

    def _update_camera_cache(
        self,
        obs: dict[str, Any],
        info: dict[str, Any],
    ) -> None:
        if not isinstance(obs, dict) or not isinstance(info, dict):
            return
        front = self._last_columnar(obs.get("front_rgb_list"))
        if front is not None:
            try:
                import numpy as np

                if hasattr(front, "detach"):
                    front = front.detach()
                if hasattr(front, "cpu"):
                    front = front.cpu()
                front_array = np.asarray(front)
                while front_array.ndim > 3 and front_array.shape[0] == 1:
                    front_array = front_array[0]
                if front_array.ndim == 4:
                    front_array = front_array[-1]
                if front_array.ndim >= 2:
                    self.front_image_shape = (
                        int(front_array.shape[0]),
                        int(front_array.shape[1]),
                    )
            except (TypeError, ValueError):
                pass
        extrinsic = self._last_columnar(
            obs.get("front_camera_extrinsic_list")
        )
        if extrinsic is not None:
            self.front_camera_extrinsic = extrinsic
        intrinsic = self._last_columnar(info.get("front_camera_intrinsic"))
        if intrinsic is not None:
            self.front_camera_intrinsic = intrinsic

    def select_target_with_point(
        self,
        *,
        available: Any,
        point_yx: tuple[int, int],
    ) -> dict[str, Any]:
        if self.front_image_shape is None:
            raise RuntimeError("Overall image shape is unavailable for pixel grounding")
        if self.front_camera_intrinsic is None or self.front_camera_extrinsic is None:
            raise RuntimeError(
                "Overall camera intrinsic/extrinsic is unavailable for pixel grounding"
            )
        y, x = point_yx
        height, width = self.front_image_shape
        if not (0 <= y < height and 0 <= x < width):
            raise ValueError(
                f"point=[{y}, {x}] is outside overall image bounds "
                f"height={height}, width={width}"
            )
        from robomme.robomme_env.utils.choice_action_mapping import (
            select_target_with_pixel,
        )

        selected = select_target_with_pixel(
            available=available,
            pixel_like=[x, y],
            intrinsic_cv=self.front_camera_intrinsic,
            extrinsic_cv=self.front_camera_extrinsic,
            image_shape=self.front_image_shape,
        )
        if selected is None or selected.get("obj") is None:
            raise ValueError(
                f"No candidate object could be matched to point=[{y}, {x}]"
            )
        return selected

    @staticmethod
    def _last(value: Any) -> Any:
        try:
            import torch

            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    return None
                return value.reshape(-1)[-1].item()
        except ImportError:
            pass
        if isinstance(value, (list, tuple)):
            return value[-1] if value else None
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                if value.size == 0:
                    return None
                return value.reshape(-1)[-1].item()
        except ImportError:
            pass
        return value

    @staticmethod
    def _flag(value: Any) -> bool:
        try:
            import torch

            if isinstance(value, torch.Tensor):
                return bool(value.reshape(-1)[0].item())
        except ImportError:
            pass
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return bool(value.reshape(-1)[0].item())
        except ImportError:
            pass
        return bool(value)

    def execute(
        self,
        solve: Callable[[], Any],
        *,
        step_callback: Callable[[Any], None] | None = None,
    ) -> ActionExecution:
        if self.planner is None:
            raise RuntimeError("OraclePlusPlannerRuntime.reset() must be called first")
        from robomme.robomme_env.utils import planner_denseStep

        collected = _run_with_bounded_dense_collection(
            self.planner,
            solve,
            max_retained_steps=self.max_retained_steps,
            planner_dense_step=planner_denseStep,
            step_callback=step_callback,
        )
        if collected == -1:
            raise RuntimeError("High-level action returned planning failure")
        batch, simulator_frame_count = collected
        obs_batch, _, terminated_batch, truncated_batch, info_batch = batch
        self._update_camera_cache(obs_batch, info_batch)
        self.env.unwrapped.evaluate()
        evaluation = self.env.unwrapped.evaluate(solve_complete_eval=True)
        terminated = bool(self._last(terminated_batch))
        truncated = bool(self._last(truncated_batch))
        reported_status = self._last(info_batch.get("status"))
        if self._flag(evaluation.get("success", False)):
            status = "success"
        elif self._flag(evaluation.get("fail", False)):
            status = "fail"
        elif truncated:
            status = "timeout"
        elif reported_status in {"success", "fail", "timeout", "error"}:
            status = reported_status
        else:
            status = "ongoing"
        info = {
            key: self._last(value) if isinstance(value, list) else value
            for key, value in info_batch.items()
        }
        info["status"] = status
        stopped = (
            terminated
            or truncated
            or status in {"success", "fail", "timeout", "error"}
        )
        return ActionExecution(
            obs_batch=obs_batch,
            info=info,
            status=status,
            stopped=stopped,
            simulator_frame_count=simulator_frame_count,
        )

    def close(self) -> None:
        env = self.env
        try:
            if env is not None:
                env.close()
        finally:
            # Match the original EnvRunner.close_env lifecycle: do not leave the
            # MPLib planner holding an environment/native scene across episodes.
            self.planner = None
            self.env = None
            self.front_camera_intrinsic = None
            self.front_camera_extrinsic = None
            self.front_image_shape = None


class BinFillActionAdapter:
    """Bind validated BinFill decisions to concrete simulator objects."""

    point_actions = ("solve_pickup",)

    def __init__(self, runtime: OraclePlusPlannerRuntime) -> None:
        self.runtime = runtime

    def execute(
        self,
        action_name: str,
        next_subgoal: str,
        point: tuple[int, int] | None,
        *,
        step_callback: Callable[[Any], None] | None = None,
    ) -> ActionExecution:
        from robomme.robomme_env.utils.subgoal_planner_func import solve_button
        from robomme.robomme_env.utils.subgoal_planner_func import solve_pickup
        from robomme.robomme_env.utils.subgoal_planner_func import (
            solve_putonto_whenhold_binspecial,
        )

        env = self.runtime.env
        planner = self.runtime.planner
        base = env.unwrapped
        if action_name == "solve_pickup":
            if point is None:
                raise ValueError("BinFill solve_pickup requires point=[y, x]")
            selected = self.runtime.select_target_with_point(
                available=base.all_cubes,
                point_yx=point,
            )
            target = selected["obj"]
            execution = self.runtime.execute(
                lambda: solve_pickup(env, planner, obj=target),
                step_callback=step_callback,
            )
            execution.info["selected_target"] = {
                key: value for key, value in selected.items() if key != "obj"
            }
            execution.info["requested_point_yx"] = list(point)
            return execution
        if point is not None:
            raise ValueError(f"{action_name} requires point=null")
        if action_name == "solve_putonto_whenhold_binspecial":
            return self.runtime.execute(
                lambda: solve_putonto_whenhold_binspecial(
                    env,
                    planner,
                    target=base.board_with_hole,
                ),
                step_callback=step_callback,
            )
        if action_name == "solve_button":
            return self.runtime.execute(
                lambda: solve_button(env, planner, obj=base.button),
                step_callback=step_callback,
            )
        raise RuntimeError(
            f"Subgoal mapper returned an unsupported BinFill action: {action_name!r}"
        )


class ButtonUnmaskActionAdapter:
    """Bind ButtonUnmask decisions to the button and visible containers."""

    point_actions = ("solve_pickup_bin",)

    def __init__(self, runtime: OraclePlusPlannerRuntime) -> None:
        self.runtime = runtime

    def execute(
        self,
        action_name: str,
        next_subgoal: str,
        point: tuple[int, int] | None,
        *,
        step_callback: Callable[[Any], None] | None = None,
    ) -> ActionExecution:
        from robomme.robomme_env.utils.subgoal_planner_func import solve_button
        from robomme.robomme_env.utils.subgoal_planner_func import solve_pickup_bin
        from robomme.robomme_env.utils.subgoal_planner_func import (
            solve_putdown_whenhold,
        )

        env = self.runtime.env
        planner = self.runtime.planner
        base = env.unwrapped
        if action_name == "solve_pickup_bin":
            if point is None:
                raise ValueError("ButtonUnmask container pickup requires point=[y, x]")
            selected = self.runtime.select_target_with_point(
                available=base.spawned_bins,
                point_yx=point,
            )
            execution = self.runtime.execute(
                lambda: solve_pickup_bin(env, planner, obj=selected["obj"]),
                step_callback=step_callback,
            )
            execution.info["selected_target"] = {
                key: value for key, value in selected.items() if key != "obj"
            }
            execution.info["requested_point_yx"] = list(point)
            return execution
        if point is not None:
            raise ValueError(f"{action_name} requires point=null")
        if action_name == "solve_button":
            button = getattr(base, "button_left", None)
            if button is None:
                button = getattr(base, "button", None)
            if button is None:
                raise RuntimeError("ButtonUnmask button object is unavailable")
            return self.runtime.execute(
                lambda: solve_button(env, planner, obj=button),
                step_callback=step_callback,
            )
        if action_name == "solve_putdown_whenhold":
            return self.runtime.execute(
                lambda: solve_putdown_whenhold(env, planner),
                step_callback=step_callback,
            )
        raise RuntimeError(
            "Subgoal mapper returned an unsupported ButtonUnmask action: "
            f"{action_name!r}"
        )


class PickHighlightActionAdapter:
    """Bind PickHighlight decisions to the button and grounded cube."""

    point_actions = ("solve_pickup",)

    def __init__(self, runtime: OraclePlusPlannerRuntime) -> None:
        self.runtime = runtime

    def execute(
        self,
        action_name: str,
        next_subgoal: str,
        point: tuple[int, int] | None,
        *,
        step_callback: Callable[[Any], None] | None = None,
    ) -> ActionExecution:
        from robomme.robomme_env.utils.subgoal_planner_func import solve_button
        from robomme.robomme_env.utils.subgoal_planner_func import solve_pickup
        from robomme.robomme_env.utils.subgoal_planner_func import (
            solve_putdown_whenhold,
        )

        env = self.runtime.env
        planner = self.runtime.planner
        base = env.unwrapped
        if action_name == "solve_pickup":
            if point is None:
                raise ValueError("PickHighlight cube pickup requires point=[y, x]")
            selected = self.runtime.select_target_with_point(
                available=base.all_cubes,
                point_yx=point,
            )
            execution = self.runtime.execute(
                lambda: solve_pickup(env, planner, obj=selected["obj"]),
                step_callback=step_callback,
            )
            execution.info["selected_target"] = {
                key: value for key, value in selected.items() if key != "obj"
            }
            execution.info["requested_point_yx"] = list(point)
            return execution
        if point is not None:
            raise ValueError(f"{action_name} requires point=null")
        if action_name == "solve_button":
            button = getattr(base, "button", None)
            if button is None:
                button = getattr(base, "button_left", None)
            if button is None:
                raise RuntimeError("PickHighlight button object is unavailable")
            return self.runtime.execute(
                lambda: solve_button(env, planner, obj=button),
                step_callback=step_callback,
            )
        if action_name == "solve_putdown_whenhold":
            return self.runtime.execute(
                lambda: solve_putdown_whenhold(env, planner, release_z=0.01),
                step_callback=step_callback,
            )
        raise RuntimeError(
            "Subgoal mapper returned an unsupported PickHighlight action: "
            f"{action_name!r}"
        )


class MoveCubeActionAdapter:
    """Execute the fixed-object MoveCube strategy inferred from its demo."""

    point_actions: tuple[str, ...] = ()

    def __init__(self, runtime: OraclePlusPlannerRuntime) -> None:
        self.runtime = runtime

    @staticmethod
    def _require(base: Any, name: str) -> Any:
        value = getattr(base, name, None)
        if value is None:
            raise RuntimeError(f"MoveCube environment attribute {name!r} is unavailable")
        return value

    def execute(
        self,
        action_name: str,
        next_subgoal: str,
        point: tuple[int, int] | None,
        *,
        step_callback: Callable[[Any], None] | None = None,
    ) -> ActionExecution:
        from robomme.robomme_env.utils.subgoal_planner_func import (
            grasp_and_lift_peg_side,
        )
        from robomme.robomme_env.utils.subgoal_planner_func import solve_pickup
        from robomme.robomme_env.utils.subgoal_planner_func import (
            solve_push_to_target,
        )
        from robomme.robomme_env.utils.subgoal_planner_func import (
            solve_push_to_target_with_peg,
        )
        from robomme.robomme_env.utils.subgoal_planner_func import (
            solve_putonto_whenhold,
        )

        if point is not None:
            raise ValueError(f"MoveCube {action_name} requires point=null")
        env = self.runtime.env
        planner = self.runtime.planner
        base = env.unwrapped
        cube = self._require(base, "cube")
        target = self._require(base, "goal_site")
        if action_name == "grasp_and_lift_peg_side":
            peg_target = self._require(base, "grasp_target")
            return self.runtime.execute(
                lambda: grasp_and_lift_peg_side(env, planner, obj=peg_target),
                step_callback=step_callback,
            )
        if action_name == "solve_push_to_target_with_peg":
            direction = self._require(base, "direction2")
            obj_flag = self._require(base, "obj_flag")
            return self.runtime.execute(
                lambda: solve_push_to_target_with_peg(
                    env,
                    planner,
                    obj=cube,
                    target=target,
                    direction=direction,
                    obj_flag=obj_flag,
                ),
                step_callback=step_callback,
            )
        if action_name == "solve_push_to_target":
            return self.runtime.execute(
                lambda: solve_push_to_target(env, planner, obj=cube, target=target),
                step_callback=step_callback,
            )
        if action_name == "solve_pickup":
            return self.runtime.execute(
                lambda: solve_pickup(env, planner, obj=cube),
                step_callback=step_callback,
            )
        if action_name == "solve_putonto_whenhold":
            return self.runtime.execute(
                lambda: solve_putonto_whenhold(env, planner, target=target),
                step_callback=step_callback,
            )
        raise RuntimeError(
            f"Subgoal mapper returned an unsupported MoveCube action: {action_name!r}"
        )


def build_action_adapter(task: str, runtime: OraclePlusPlannerRuntime) -> Any:
    if task == "BinFill":
        return BinFillActionAdapter(runtime)
    if task == "ButtonUnmask":
        return ButtonUnmaskActionAdapter(runtime)
    if task == "PickHighlight":
        return PickHighlightActionAdapter(runtime)
    if task == "MoveCube":
        return MoveCubeActionAdapter(runtime)
    raise ValueError(
        f"Oracle Plus has no direct high-level action adapter for task {task!r}"
    )

