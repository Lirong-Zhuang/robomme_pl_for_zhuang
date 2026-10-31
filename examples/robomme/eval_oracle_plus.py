"""Evaluate Oracle Plus through a remote Qwen3.8-27B WebSocket service.

Run from the repository root:
    bash scripts/eval_oracle_plus.sh

Direct invocation inside the server environment:
    micromamba run -n robomme python examples/robomme/eval_oracle_plus.py \
        --vlm-host 127.0.0.1 --vlm-port 8021 \
        --tasks BinFill --num-episodes 10 --num-repeats 3 \
        --run-name oracle_plus_27b.1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party" / "robomme_benchmark" / "src"
for import_root in (REPO_ROOT / "src", THIRD_PARTY_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from oracle_plus.action_runtime import OraclePlusPlannerRuntime
from oracle_plus.action_runtime import build_action_adapter
from oracle_plus.prompts import available_tasks
from oracle_plus.prompts import build_messages
from oracle_plus.prompts import get_point_action_names
from oracle_plus.prompts import get_prompt
from oracle_plus.prompts import resolve_action_name
from oracle_plus.runtime_metrics import GpuMemoryMonitor
from oracle_plus.runtime_metrics import latency_summary
from oracle_plus.vlm import SubgoalDecision
from oracle_plus.vlm_client import VLMSubgoalPlannerClient
from robomme.env_record_wrapper import BenchmarkEnvBuilder


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _episode_ids(value: str) -> list[int] | None:
    if not value.strip():
        return None
    result = [int(item) for item in _csv(value)]
    if any(item < 0 for item in result):
        raise ValueError("Episode IDs must be non-negative")
    if len(result) != len(set(result)):
        raise ValueError("Episode IDs must not contain duplicates")
    return result


def _safe_name(value: str) -> str:
    candidate = value.strip()
    if not candidate or candidate in {".", ".."} or Path(candidate).name != candidate:
        raise argparse.ArgumentTypeError("run name must be one directory name")
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-host", default="127.0.0.1")
    parser.add_argument("--vlm-port", type=int, default=8021)
    parser.add_argument("--tasks", default="BinFill", help="Comma-separated task names.")
    parser.add_argument(
        "--episode-ids",
        default="",
        help="Comma-separated IDs; overrides --num-episodes when non-empty.",
    )
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=1,
        help="Repeat the same fixed benchmark episodes this many times.",
    )
    parser.add_argument("--max-steps", type=int, default=1300)
    parser.add_argument("--max-decisions", type=int, default=20)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--save-dir", type=Path, default=Path("runs/evaluation/oracle-plus"))
    parser.add_argument("--run-name", type=_safe_name, default="oracle_plus_27b.1")
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow writing into an existing run directory.",
    )
    parser.add_argument(
        "--include-description",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the prompt/output schema with a required visual description; "
            "--no-include-description selects the compact schema."
        ),
    )
    return parser.parse_args()


def _to_rgb(value: Any) -> np.ndarray:
    while isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Cannot convert an empty image sequence")
        value = value[-1]
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    image = np.asarray(value)
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 4:
        image = image[-1]
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"Expected HWC RGB image, found shape {image.shape}")
    image = image[..., :3]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.size and image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _current_pair(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return _to_rgb(obs["front_rgb_list"]), _to_rgb(obs["wrist_rgb_list"])


def _recent_pairs(
    obs_batch: dict[str, list[Any]],
    limit: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    fronts = obs_batch.get("front_rgb_list", [])
    wrists = obs_batch.get("wrist_rgb_list", [])
    if not isinstance(fronts, list):
        fronts = [fronts]
    if not isinstance(wrists, list):
        wrists = [wrists]
    count = min(len(fronts), len(wrists))
    if count == 0:
        return []
    indices = list(range(max(0, count - limit), count))
    return [(_to_rgb(fronts[index]), _to_rgb(wrists[index])) for index in indices]


def _write_pair(
    directory: Path,
    prefix: str,
    pair: tuple[np.ndarray, np.ndarray],
) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    overall_path = directory / f"{prefix}_overall.png"
    wrist_path = directory / f"{prefix}_wrist.png"
    iio.imwrite(overall_path, pair[0])
    iio.imwrite(wrist_path, pair[1])
    return [str(overall_path.resolve()), str(wrist_path.resolve())]


def _write_observation_request(
    episode_dir: Path,
    decision_index: int,
    before_pair: tuple[np.ndarray, np.ndarray],
    recent_pairs: list[tuple[np.ndarray, np.ndarray]],
) -> list[str]:
    request_dir = episode_dir / f"decision_{decision_index:03d}"
    paths = _write_pair(request_dir, "before", before_pair)
    for index, pair in enumerate(recent_pairs, start=1):
        paths.extend(_write_pair(request_dir, f"recent_{index:02d}", pair))
    return paths


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except ImportError:
        pass
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _task_goal_candidates(info: dict[str, Any]) -> list[str]:
    value = info.get("task_goal", "")
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple)):
        candidates = list(value)
    else:
        raise ValueError(f"Environment returned an invalid task_goal: {value!r}")
    if not candidates or not all(
        isinstance(candidate, str) and candidate.strip()
        for candidate in candidates
    ):
        raise ValueError(f"Environment returned an invalid task_goal: {value!r}")
    return [candidate.strip() for candidate in candidates]


def _task_goal(info: dict[str, Any]) -> str:
    """Use RoboMME's first goal paraphrase, matching the original EnvRunner."""
    return _task_goal_candidates(info)[0]


def _infer(
    planner: VLMSubgoalPlannerClient,
    *,
    messages: list[dict[str, str]],
    images: list[str],
    include_description: bool,
) -> tuple[SubgoalDecision, str, float, float]:
    return planner.infer(
        messages=messages,
        images=images,
        include_description=include_description,
    )


def _resolve_decision_action(task: str, decision: SubgoalDecision) -> str | None:
    for planned_subgoal in decision.subgoal_sequence:
        resolve_action_name(task, planned_subgoal)
    action_name = resolve_action_name(task, decision.next_subgoal)
    if action_name is None:
        if decision.point is not None:
            raise ValueError("STOP requires point=null")
        return None
    requires_point = action_name in get_point_action_names(task)
    if requires_point and decision.point is None:
        raise ValueError(
            f"Subgoal {decision.next_subgoal!r} resolves to {action_name}, "
            "which requires point=[y, x]"
        )
    if not requires_point and decision.point is not None:
        raise ValueError(
            f"Subgoal {decision.next_subgoal!r} resolves to {action_name}, "
            "which requires point=null"
        )
    return action_name


def run_episode(
    *,
    task: str,
    episode_id: int,
    repeat_id: int,
    args: argparse.Namespace,
    model: VLMSubgoalPlannerClient,
    gpu_monitor: GpuMemoryMonitor,
    run_dir: Path,
) -> dict[str, Any]:
    builder = BenchmarkEnvBuilder(
        env_id=task,
        dataset="test",
        action_space="joint_angle",
        gui_render=False,
        max_steps=args.max_steps,
    )
    env = builder.make_env_for_episode(
        episode_id,
        max_steps=args.max_steps,
        include_front_camera_extrinsic=True,
        include_front_camera_intrinsic=True,
    )
    runtime = OraclePlusPlannerRuntime(env, task, gui_render=False)
    episode_dir = (
        run_dir
        / task
        / f"repeat_{repeat_id}"
        / f"episode_{episode_id:04d}"
    )
    episode_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "task": task,
        "episode_id": episode_id,
        "repeat_id": repeat_id,
        "task_goal": None,
        "task_goal_candidates": [],
        "status": "error",
        "decisions": [],
        "prompt": get_prompt(
            task,
            include_description=args.include_description,
        ).metadata(),
    }
    try:
        obs, reset_info = runtime.reset()
        goal_candidates = _task_goal_candidates(reset_info)
        goal = _task_goal(reset_info)
        record["task_goal"] = goal
        record["task_goal_candidates"] = goal_candidates
        adapter = build_action_adapter(task, runtime)
        if tuple(adapter.point_actions) != get_point_action_names(task):
            raise RuntimeError(
                f"{task} target-point registries do not match: "
                f"prompt={get_point_action_names(task)}, "
                f"adapter={adapter.point_actions}"
            )
        current_pair = _current_pair(obs)
        initial_paths = _write_observation_request(
            episode_dir,
            decision_index=0,
            before_pair=current_pair,
            recent_pairs=[],
        )
        messages = build_messages(
            task=task,
            call_type="initial",
            task_goal=goal,
            include_description=args.include_description,
            overall_image_shape=current_pair[0].shape[:2],
        )
        decision, raw_response, latency_ms, request_to_decision_ms = _infer(
            model,
            messages=messages,
            images=initial_paths,
            include_description=args.include_description,
        )
        mapping_started = time.perf_counter()
        resolved_action_name = _resolve_decision_action(task, decision)
        action_request_ms = (
            request_to_decision_ms
            + (time.perf_counter() - mapping_started) * 1000
        )
        gpu_monitor.sample()
        record["decisions"].append(
            {
                "decision_index": 0,
                "call_type": "initial",
                "messages": messages,
                "images": initial_paths,
                "raw_response": raw_response,
                "decision": asdict(decision),
                "resolved_action_name": resolved_action_name,
                "vlm_latency_ms": latency_ms,
                "action_request_ms": action_request_ms,
                "execution": None,
            }
        )
        history: list[str] = []
        last_status = "ongoing"

        for decision_index in range(1, args.max_decisions + 1):
            if decision.is_stop:
                record["status"] = (
                    "success" if last_status == "success" else "invalid_early_stop"
                )
                break
            if resolved_action_name is None:
                raise RuntimeError("Non-STOP subgoal unexpectedly resolved to no action")

            active_subgoal = decision.next_subgoal
            execution = adapter.execute(
                resolved_action_name,
                active_subgoal,
                decision.point,
            )
            gpu_monitor.sample()
            last_status = execution.status
            recent_pairs = _recent_pairs(execution.obs_batch, args.history_size)
            if not recent_pairs:
                recent_pairs = [current_pair]
            current_pair_after = recent_pairs[-1]
            record["decisions"][-1]["execution"] = {
                "resolved_action_name": resolved_action_name,
                "next_subgoal": active_subgoal,
                "point": decision.point,
                "status": execution.status,
                "stopped": execution.stopped,
                "info": _jsonable(execution.info),
                "simulator_frame_count": len(
                    execution.obs_batch.get("front_rgb_list", [])
                ),
            }
            if execution.stopped:
                record["status"] = execution.status
                break

            image_paths = _write_observation_request(
                episode_dir,
                decision_index=decision_index,
                before_pair=current_pair,
                recent_pairs=recent_pairs,
            )
            messages = build_messages(
                task=task,
                call_type="followup",
                task_goal=goal,
                include_description=args.include_description,
                history_subgoals=history,
                current_subgoal=active_subgoal,
                observation_count=len(recent_pairs),
                overall_image_shape=current_pair_after[0].shape[:2],
            )
            (
                next_decision,
                raw_response,
                latency_ms,
                request_to_decision_ms,
            ) = _infer(
                model,
                messages=messages,
                images=image_paths,
                include_description=args.include_description,
            )
            mapping_started = time.perf_counter()
            next_resolved_action_name = _resolve_decision_action(task, next_decision)
            action_request_ms = (
                request_to_decision_ms
                + (time.perf_counter() - mapping_started) * 1000
            )
            gpu_monitor.sample()
            if next_decision.completed:
                history.append(active_subgoal)
            record["decisions"].append(
                {
                    "decision_index": decision_index,
                    "call_type": "followup",
                    "messages": messages,
                    "images": image_paths,
                    "raw_response": raw_response,
                    "decision": asdict(next_decision),
                    "resolved_action_name": next_resolved_action_name,
                    "vlm_latency_ms": latency_ms,
                    "action_request_ms": action_request_ms,
                    "execution": None,
                }
            )
            decision = next_decision
            resolved_action_name = next_resolved_action_name
            current_pair = current_pair_after
        else:
            record["status"] = "decision_limit"
    except Exception as error:
        record["status"] = "error"
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc()
    finally:
        runtime.close()

    output_path = episode_dir / "episode.json"
    output_path.write_text(
        json.dumps(_jsonable(record), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return record


def main() -> None:
    args = parse_args()
    tasks = _csv(args.tasks)
    if not tasks:
        raise ValueError("--tasks must contain at least one task")
    unsupported = [task for task in tasks if task not in available_tasks()]
    if unsupported:
        raise ValueError(
            f"Oracle Plus currently lacks task-specific prompts/adapters for {unsupported}; "
            f"available={available_tasks()}"
        )
    if (
        args.num_episodes < 1
        or args.num_repeats < 1
        or args.max_decisions < 1
        or args.history_size < 1
    ):
        raise ValueError(
            "Episode, repeat, decision, and history counts must all be positive"
        )
    selected_episode_ids = _episode_ids(args.episode_ids)
    episode_ids = selected_episode_ids or list(range(args.num_episodes))

    run_dir = (args.save_dir / args.run_name).expanduser().resolve()
    if run_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Use --overwrite to continue."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Oracle Plus tasks: {tasks}; episodes: {episode_ids}; "
        f"repeats: {args.num_repeats}"
    )
    print(f"Connecting to Qwen3.8-27B at {args.vlm_host}:{args.vlm_port}")
    eval_gpu_monitor = GpuMemoryMonitor(
        os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    )
    model = VLMSubgoalPlannerClient(args.vlm_host, args.vlm_port)
    if model.server_metadata.get("protocol") != "oracle_plus_vlm_v1":
        raise RuntimeError(
            "Connected server does not provide the Oracle Plus VLM protocol: "
            f"{model.server_metadata!r}"
        )

    records: list[dict[str, Any]] = []
    predictions_path = run_dir / "episodes.jsonl"
    with predictions_path.open("w", encoding="utf-8") as output:
        for repeat_id in range(1, args.num_repeats + 1):
            for task in tasks:
                for episode_id in episode_ids:
                    print(
                        f"[repeat {repeat_id}/{args.num_repeats}] "
                        f"[{task}] episode {episode_id} ...",
                        flush=True,
                    )
                    record = run_episode(
                        task=task,
                        episode_id=episode_id,
                        repeat_id=repeat_id,
                        args=args,
                        model=model,
                        gpu_monitor=eval_gpu_monitor,
                        run_dir=run_dir,
                    )
                    records.append(record)
                    output.write(
                        json.dumps(_jsonable(record), ensure_ascii=False) + "\n"
                    )
                    output.flush()
                    print(
                        f"[repeat {repeat_id}/{args.num_repeats}] "
                        f"[{task}] episode {episode_id}: {record['status']}",
                        flush=True,
                    )

    successes = sum(record["status"] == "success" for record in records)
    vlm_step_times_ms = [
        float(decision["vlm_latency_ms"])
        for record in records
        for decision in record["decisions"]
    ]
    action_step_times_ms = [
        float(decision["action_request_ms"])
        for record in records
        for decision in record["decisions"]
    ]
    repeat_summaries = []
    for repeat_id in range(1, args.num_repeats + 1):
        repeat_records = [
            record for record in records if record["repeat_id"] == repeat_id
        ]
        repeat_successes = sum(
            record["status"] == "success" for record in repeat_records
        )
        repeat_summaries.append(
            {
                "repeat_id": repeat_id,
                "episode_runs": len(repeat_records),
                "successes": repeat_successes,
                "success_rate": (
                    repeat_successes / len(repeat_records)
                    if repeat_records
                    else 0.0
                ),
                "status_counts": {
                    status: sum(
                        record["status"] == status for record in repeat_records
                    )
                    for status in sorted(
                        {record["status"] for record in repeat_records}
                    )
                },
            }
        )
    vlm_metrics = model.metrics()
    eval_gpu_monitor.sample()
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "framework": "oracle_plus",
        "model_family": vlm_metrics["model_family"],
        "model_path": vlm_metrics["model_path"],
        "environments": {
            "vlm_server": "robomme-qwen38",
            "evaluation": "robomme",
        },
        "vlm_endpoint": {
            "host": args.vlm_host,
            "port": args.vlm_port,
            "protocol": vlm_metrics["protocol"],
        },
        "tasks": tasks,
        "episode_ids": episode_ids,
        "num_repeats": args.num_repeats,
        "episode_runs_per_repeat": len(tasks) * len(episode_ids),
        "episode_runs": len(records),
        "episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records) if records else 0.0,
        "status_counts": {
            status: sum(record["status"] == status for record in records)
            for status in sorted({record["status"] for record in records})
        },
        "repeat_summaries": repeat_summaries,
        "inference_config": {
            **vlm_metrics["inference_config"],
            "include_description": args.include_description,
        },
        "gpu_and_timing": {
            "gpu": {
                "vlm_server_process": vlm_metrics["gpu"],
                "eval_process": eval_gpu_monitor.summary(),
            },
            "model_load_seconds": vlm_metrics["model_load_seconds"],
            "vlm_single_step": latency_summary(
                vlm_step_times_ms,
                scope=(
                    "One Qwen3.8-27B engine.infer call, timed after InferRequest "
                    "construction until raw model output returns inside the VLM "
                    "server. Image saving, WebSocket transfer, path transfer, "
                    "and request unpacking are excluded; batch size is 1."
                ),
            ),
            "action_single_step": latency_summary(
                action_step_times_ms,
                scope=(
                    "One action request from handing the prepared image request "
                    "to the VLM through JSON validation and subgoal-to-action "
                    "mapping. Robot/MPLib execution is excluded."
                ),
            ),
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    model.close()
    eval_gpu_monitor.close()


if __name__ == "__main__":
    main()
