"""Generate RoboMME HDF5 episodes with an intentional failed grasp.

SERVER QUICK START
==================

Run every command below from the repository root on the server. Everything is
written directly to the server path configured in this file.

1. Prepare the RoboMME simulator environment once::

    git submodule update --init
    uv sync --project third_party/robomme_benchmark

2. Edit ``SERVER_OUTPUT_DIR`` in the USER CONFIG block below. Use one absolute,
   persistent server path, for example::

    SERVER_OUTPUT_DIR = "/data/zhuang/binfill_failure_recovery"

   All generated HDF5 episodes go into this one raw-data pool. Do not divide
   HDF5 into train/test directories. Make an episode-disjoint split only when
   building the later Reporter JSONL/index.

3. Generate one smoke episode and inspect its FailRecoverZ video::

    uv run --project third_party/robomme_benchmark \
      python scripts/generate_failure_recovery_h5.py \
      --task BinFill --failure-mode z --difficulty easy \
      --num-episodes 1 --start-episode 0 --base-seed 100000 --save-video

4. Define one reusable server-side batch command::

    generate_batch () {
      MODE=$1
      DIFFICULTY=$2
      COUNT=$3
      START_EPISODE=$4
      BASE_SEED=$5

      uv run --project third_party/robomme_benchmark \
        python scripts/generate_failure_recovery_h5.py \
        --task BinFill --failure-mode "$MODE" --difficulty "$DIFFICULTY" \
        --num-episodes "$COUNT" --start-episode "$START_EPISODE" \
        --base-seed "$BASE_SEED" --save-video
    }

5. After the smoke episode looks correct, generate the remaining 29 episodes.
   Together these commands create one 30-episode pool (15 z, 15 xy; 15 easy,
   8 medium, 7 hard). Episode 0 is the smoke episode from step 3::

    generate_batch z  easy   7  1  110000
    generate_batch z  medium 4  8  120000
    generate_batch z  hard   3 12  130000
    generate_batch xy easy   7 15  200000
    generate_batch xy medium 4 22  210000
    generate_batch xy hard   4 26  220000

For long server runs, execute the same commands inside tmux. The accepted HDF5
files are all stored in ``<SERVER_OUTPUT_DIR>/hdf5_files``. Videos are in
``<SERVER_OUTPUT_DIR>/videos``. Per-batch manifests such as
``generation_manifest_z_easy_ep1_to_7.json`` preserve the failure mode,
difficulty, episode number, accepted seed, and HDF5 path. Re-running an
existing task/episode/seed is refused unless ``--overwrite`` is passed.

The simulator first executes a deliberately offset grasp (``z`` or ``xy``),
then continues with the normal oracle pickup so the complete episode can still
succeed and be persisted by ``RobommeRecordWrapper``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import random
import sys
from typing import Any


# =============================== USER CONFIG ===============================
# Set this once on the server. It must be an absolute, persistent directory.
# Example: SERVER_OUTPUT_DIR = "/data/zhuang/binfill_failure_recovery"
# A command-line --output-dir may still be used as a temporary override.
SERVER_OUTPUT_DIR = "/data/zhuanglr/robomme/data/h5_data"
# ===========================================================================


REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOMME_ROOT = REPO_ROOT / "third_party" / "robomme_benchmark"
ROBOMME_SRC = ROBOMME_ROOT / "src"

SUPPORTED_TASKS = (
    "BinFill",
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "PickXtimes",
    "SwingXtimes",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "VideoRepick",
    "VideoUnmask",
    "VideoUnmaskSwap",
)
FAILURE_MODES = ("z", "xy")
DIFFICULTIES = ("easy", "medium", "hard")


@dataclass(frozen=True)
class Runtime:
    gym: Any
    h5py: Any
    numpy: Any
    torch: Any
    record_wrapper: type
    failsafe_timeout: type[Exception]
    scene_generation_error: type[Exception]
    arm_planner: type
    screw_plan_failure: type[Exception]


@dataclass(frozen=True)
class EpisodeResult:
    task: str
    episode: int
    seed: int
    difficulty: str
    failure_mode: str
    h5_path: str
    video_enabled: bool
    attempts: int


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate successful RoboMME oracle episodes containing an "
            "intentional failed grasp followed by a successful retry."
        )
    )
    parser.add_argument("--task", choices=SUPPORTED_TASKS, default="BinFill")
    parser.add_argument(
        "--failure-mode",
        choices=FAILURE_MODES,
        required=True,
        help="z stops above the object; xy offsets the grasp laterally.",
    )
    parser.add_argument(
        "--difficulty",
        choices=DIFFICULTIES,
        default="easy",
    )
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="First seed block. Each logical episode owns a disjoint retry block.",
    )
    parser.add_argument("--max-seed-attempts", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(SERVER_OUTPUT_DIR).expanduser() if SERVER_OUTPUT_DIR else None,
        help=(
            "Raw output pool. Defaults to SERVER_OUTPUT_DIR near the top of "
            "this file. All failure modes and difficulties share this path."
        ),
    )
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save rollout videos for visual verification (default: true).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing HDF5 with the same task, episode, and seed.",
    )
    args = parser.parse_args(argv)
    if args.num_episodes < 1:
        parser.error("--num-episodes must be at least 1")
    if args.start_episode < 0:
        parser.error("--start-episode must be non-negative")
    if args.max_seed_attempts < 1:
        parser.error("--max-seed-attempts must be at least 1")
    if args.output_dir is None:
        parser.error(
            "Set SERVER_OUTPUT_DIR near the top of this file or pass "
            "--output-dir /absolute/server/path"
        )
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def _candidate_seed(
    base_seed: int,
    logical_episode_offset: int,
    attempt: int,
    max_seed_attempts: int,
) -> int:
    return base_seed + logical_episode_offset * max_seed_attempts + attempt


def _candidate_h5_path(
    output_dir: Path,
    task: str,
    episode: int,
    seed: int,
) -> Path:
    return output_dir / "hdf5_files" / f"{task}_ep{episode}_seed{seed}.h5"


def _load_runtime() -> Runtime:
    if not ROBOMME_SRC.is_dir():
        raise RuntimeError(
            "RoboMME submodule is missing. Run `git submodule update --init`."
        )
    src = str(ROBOMME_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)

    try:
        import gymnasium as gym
        import h5py
        import numpy as np
        import torch

        from robomme.env_record_wrapper import FailsafeTimeout
        from robomme.env_record_wrapper import RobommeRecordWrapper
        from robomme.robomme_env.utils.SceneGenerationError import (
            SceneGenerationError,
        )
        from robomme.robomme_env.utils.planner_fail_safe import (
            FailAwarePandaArmMotionPlanningSolver,
        )
        from robomme.robomme_env.utils.planner_fail_safe import ScrewPlanFailure
    except ImportError as error:
        raise RuntimeError(
            "RoboMME simulator dependencies are unavailable. Run with its "
            "environment, for example `uv run --project "
            "third_party/robomme_benchmark python "
            "scripts/generate_failure_recovery_h5.py ...`."
        ) from error

    # Importing the module registers all custom Gym environments.
    importlib.import_module("robomme.robomme_env")
    return Runtime(
        gym=gym,
        h5py=h5py,
        numpy=np,
        torch=torch,
        record_wrapper=RobommeRecordWrapper,
        failsafe_timeout=FailsafeTimeout,
        scene_generation_error=SceneGenerationError,
        arm_planner=FailAwarePandaArmMotionPlanningSolver,
        screw_plan_failure=ScrewPlanFailure,
    )


def _tensor_to_bool(value: Any, torch_module: Any, numpy_module: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, torch_module.Tensor):
        return bool(value.detach().cpu().bool().item())
    if isinstance(value, numpy_module.ndarray):
        return bool(numpy_module.any(value))
    return bool(value)


def _patch_planner_screw_to_rrt(
    planner: Any,
    screw_plan_failure: type[Exception],
    max_attempts: int = 3,
) -> None:
    """Retry screw planning, then fall back to RRT*, matching upstream tests."""
    original_screw = planner.move_to_pose_with_screw
    original_rrt = planner.move_to_pose_with_RRTStar

    def _move_screw_then_rrt(*args: Any, **kwargs: Any) -> Any:
        for _ in range(max_attempts):
            try:
                result = original_screw(*args, **kwargs)
            except screw_plan_failure:
                continue
            if not (isinstance(result, int) and result == -1):
                return result

        for _ in range(max_attempts):
            try:
                result = original_rrt(*args, **kwargs)
            except Exception:
                continue
            if not (isinstance(result, int) and result == -1):
                return result
        return -1

    planner.move_to_pose_with_screw = _move_screw_then_rrt


def _validate_written_episode(
    h5_path: Path,
    episode: int,
    h5py_module: Any,
) -> None:
    if not h5_path.is_file():
        raise RuntimeError(f"Recorder did not create expected HDF5: {h5_path}")
    episode_key = f"episode_{episode}"
    with h5py_module.File(h5_path, "r") as data:
        if episode_key not in data:
            raise RuntimeError(
                f"Successful rollout did not contain {episode_key!r}: {h5_path}"
            )
        timestep_count = sum(
            key.startswith("timestep_") for key in data[episode_key]
        )
        if timestep_count == 0:
            raise RuntimeError(f"Recorded episode contains no timesteps: {h5_path}")


def _run_one_episode(
    runtime: Runtime,
    *,
    task: str,
    episode: int,
    seed: int,
    difficulty: str,
    failure_mode: str,
    output_dir: Path,
    save_video: bool,
) -> tuple[bool, str]:
    # RoboMME owns an environment-local torch generator, while the upstream
    # xy failure perturbation also uses Python's global ``random`` module.
    # Seed all three sources so a recorded failure can be reproduced.
    random.seed(seed)
    runtime.numpy.random.seed(seed)
    runtime.torch.manual_seed(seed)

    env_kwargs = {
        "obs_mode": "rgb+depth+segmentation",
        "control_mode": "pd_joint_pos",
        "render_mode": "rgb_array",
        "reward_mode": "dense",
        "seed": seed,
        "difficulty": difficulty,
        "robomme_failure_recovery": True,
        "robomme_failure_recovery_mode": failure_mode,
    }

    env = None
    planner = None
    try:
        env = runtime.gym.make(task, **env_kwargs)
        env = runtime.record_wrapper(
            env,
            dataset=str(output_dir),
            env_id=task,
            episode=episode,
            seed=seed,
            save_video=save_video,
        )
        env.reset()

        injected_task_index = getattr(
            env.unwrapped,
            "fail_grasp_task_index",
            None,
        )
        if injected_task_index is None:
            raise RuntimeError(
                f"{task} did not inject a failed grasp. The task may not expose "
                "a supported pickup subgoal in this configuration."
            )

        planner = runtime.arm_planner(
            env,
            debug=False,
            vis=False,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
        )
        _patch_planner_screw_to_rrt(planner, runtime.screw_plan_failure)

        episode_successful = False
        tasks = list(getattr(env.unwrapped, "task_list", []) or [])
        for task_entry in tasks:
            solve_callable = task_entry.get("solve")
            if not callable(solve_callable):
                continue
            env.unwrapped.evaluate(solve_complete_eval=True)
            screw_failed = False
            try:
                solve_result = solve_callable(env, planner)
                if isinstance(solve_result, int) and solve_result == -1:
                    screw_failed = True
            except runtime.screw_plan_failure:
                screw_failed = True
            except runtime.failsafe_timeout:
                return False, "failsafe timeout"

            if screw_failed:
                env.unwrapped.failureflag = runtime.torch.tensor([True])
                env.unwrapped.successflag = runtime.torch.tensor([False])
                env.unwrapped.current_task_failure = True

            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
            if _tensor_to_bool(
                evaluation.get("success"),
                runtime.torch,
                runtime.numpy,
            ):
                episode_successful = True
                break
            if screw_failed or _tensor_to_bool(
                evaluation.get("fail"),
                runtime.torch,
                runtime.numpy,
            ):
                break
        else:
            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
            episode_successful = _tensor_to_bool(
                evaluation.get("success"),
                runtime.torch,
                runtime.numpy,
            )

        episode_successful = episode_successful or _tensor_to_bool(
            getattr(env, "episode_success", False),
            runtime.torch,
            runtime.numpy,
        )
        if episode_successful and not bool(
            getattr(env.unwrapped, "use_fail_planner", False)
        ):
            return False, "episode succeeded without executing the injected grasp"
        return episode_successful, "success" if episode_successful else "task failed"
    except runtime.scene_generation_error as error:
        return False, f"scene generation failed: {error}"
    finally:
        if planner is not None:
            try:
                planner.close()
            except Exception:
                pass
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _generate(args: argparse.Namespace) -> list[EpisodeResult]:
    runtime = _load_runtime()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[EpisodeResult] = []

    for offset in range(args.num_episodes):
        episode = args.start_episode + offset
        last_reason = "not attempted"
        for attempt in range(args.max_seed_attempts):
            seed = _candidate_seed(
                args.base_seed,
                offset,
                attempt,
                args.max_seed_attempts,
            )
            print(
                f"[{args.task}] episode={episode} mode={args.failure_mode} "
                f"seed={seed} attempt={attempt + 1}/{args.max_seed_attempts}"
            )
            h5_path = _candidate_h5_path(
                output_dir,
                args.task,
                episode,
                seed,
            )
            if h5_path.exists():
                if not args.overwrite:
                    raise FileExistsError(
                        f"Refusing to replace existing output: {h5_path}. "
                        "Pass --overwrite to regenerate it."
                    )
                h5_path.unlink()
            try:
                successful, reason = _run_one_episode(
                    runtime,
                    task=args.task,
                    episode=episode,
                    seed=seed,
                    difficulty=args.difficulty,
                    failure_mode=args.failure_mode,
                    output_dir=output_dir,
                    save_video=args.save_video,
                )
            except Exception as error:
                successful = False
                reason = f"{type(error).__name__}: {error}"
            last_reason = reason
            if not successful:
                # RecordWrapper leaves an empty HDF5 container for rejected
                # attempts. Remove only this exact generated candidate so a
                # later merge cannot mistake it for a usable episode.
                h5_path.unlink(missing_ok=True)
                print(f"  rejected: {reason}")
                continue

            _validate_written_episode(h5_path, episode, runtime.h5py)
            result = EpisodeResult(
                task=args.task,
                episode=episode,
                seed=seed,
                difficulty=args.difficulty,
                failure_mode=args.failure_mode,
                h5_path=str(h5_path),
                video_enabled=args.save_video,
                attempts=attempt + 1,
            )
            results.append(result)
            print(f"  accepted: {h5_path}")
            break
        else:
            raise RuntimeError(
                f"[{args.task}] episode {episode} failed after "
                f"{args.max_seed_attempts} seeds; last reason: {last_reason}"
            )

    final_episode = args.start_episode + args.num_episodes - 1
    manifest_path = output_dir / (
        f"generation_manifest_{args.failure_mode}_{args.difficulty}_"
        f"ep{args.start_episode}_to_{final_episode}.json"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "task": args.task,
                "failure_mode": args.failure_mode,
                "difficulty": args.difficulty,
                "episodes": [asdict(result) for result in results],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")
    return results


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
