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
import gc
import json
import os
import shutil
import sys
import time
import traceback
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party" / "robomme_benchmark" / "src"
ROBOMME_EXAMPLES_ROOT = REPO_ROOT / "examples" / "robomme"
for import_root in (REPO_ROOT / "src", THIRD_PARTY_ROOT, ROBOMME_EXAMPLES_ROOT):
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
from oracle_plus.runtime_metrics import cgroup_memory_snapshot
from oracle_plus.runtime_metrics import latency_summary
from oracle_plus.runtime_metrics import process_memory_snapshot
from oracle_plus.vlm import SubgoalDecision
from oracle_plus.vlm_client import VLMSubgoalPlannerClient
from robomme.env_record_wrapper import BenchmarkEnvBuilder
from utils import RolloutRecorder


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
        help=(
            "Delete an existing run and start again. With --no-overwrite, an "
            "existing partial run is resumed and completed episodes are skipped."
        ),
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
    episode_dir: Path,
    step_index: int,
    pair: tuple[np.ndarray, np.ndarray],
) -> list[str]:
    overall_dir = episode_dir / "overall"
    wrist_dir = episode_dir / "wrist"
    overall_dir.mkdir(parents=True, exist_ok=True)
    wrist_dir.mkdir(parents=True, exist_ok=True)
    overall_path = overall_dir / f"step_{step_index}.png"
    wrist_path = wrist_dir / f"step_{step_index}.png"
    iio.imwrite(overall_path, pair[0])
    iio.imwrite(wrist_path, pair[1])
    return [str(overall_path.resolve()), str(wrist_path.resolve())]


def _write_observation_steps(
    episode_dir: Path,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    start_step: int,
) -> tuple[list[list[str]], int]:
    paths_by_step = [
        _write_pair(episode_dir, start_step + offset, pair)
        for offset, pair in enumerate(pairs)
    ]
    return paths_by_step, start_step + len(pairs)


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


EpisodeKey = tuple[int, str, int]


def _episode_key(record: dict[str, Any]) -> EpisodeKey:
    """Return the stable identity of one repeat/task/episode record."""

    repeat_id = record.get("repeat_id")
    task = record.get("task")
    episode_id = record.get("episode_id")
    if (
        isinstance(repeat_id, bool)
        or not isinstance(repeat_id, int)
        or repeat_id < 1
        or not isinstance(task, str)
        or not task
        or isinstance(episode_id, bool)
        or not isinstance(episode_id, int)
        or episode_id < 0
    ):
        raise ValueError(f"Invalid Oracle Plus episode record identity: {record!r}")
    return repeat_id, task, episode_id


def _load_episode_records(path: Path) -> dict[EpisodeKey, dict[str, Any]]:
    """Load the latest canonical record for every completed/attempted episode."""

    if not path.exists():
        return {}
    records: dict[EpisodeKey, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as source:
        lines = source.readlines()
        last_nonempty_line = max(
            (index for index, line in enumerate(lines, start=1) if line.strip()),
            default=0,
        )
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                if line_number == last_nonempty_line:
                    print(
                        f"[oracle_plus] Ignoring interrupted trailing record in "
                        f"{path} line {line_number}: {error}",
                        file=sys.stderr,
                    )
                    break
                raise ValueError(
                    f"Cannot resume from malformed {path} line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Cannot resume from non-object {path} line {line_number}"
                )
            if not isinstance(record.get("status"), str):
                raise ValueError(
                    f"Cannot resume from {path} line {line_number}: "
                    "record status must be text"
                )
            records[_episode_key(record)] = record
    return records


def _ordered_episode_records(
    records: dict[EpisodeKey, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [records[key] for key in sorted(records)]


def _write_episode_records(
    path: Path,
    records: dict[EpisodeKey, dict[str, Any]],
) -> None:
    """Atomically persist one latest record per repeat/task/episode."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        for record in _ordered_episode_records(records):
            output.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
    os.replace(temporary_path, path)


def _record_is_complete(record: dict[str, Any]) -> bool:
    """Match the original eval: retry only records explicitly marked error."""

    return record.get("status") != "error"


def _progress_from_records(
    records: dict[EpisodeKey, dict[str, Any]],
) -> dict[str, dict[str, dict[str, bool | str]]]:
    progress: dict[str, dict[str, dict[str, bool | str]]] = {}
    for (repeat_id, task, episode_id), record in sorted(records.items()):
        repeat_progress = progress.setdefault(f"repeat_{repeat_id}", {})
        task_progress = repeat_progress.setdefault(task, {})
        status = record.get("status")
        task_progress[str(episode_id)] = True if status == "success" else str(status)
    return progress


def _requested_run_config(
    *,
    args: argparse.Namespace,
    tasks: list[str],
    episode_ids: list[int],
    server_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build the behavior-defining configuration used to validate a resume."""

    return {
        "schema_version": 1,
        "tasks": tasks,
        "episode_ids": episode_ids,
        "num_repeats": args.num_repeats,
        "max_steps": args.max_steps,
        "max_decisions": args.max_decisions,
        "history_size": args.history_size,
        "include_description": args.include_description,
        "prompts": {
            task: get_prompt(
                task,
                include_description=args.include_description,
            ).metadata()
            for task in tasks
        },
        "model": {
            "protocol": server_metadata.get("protocol"),
            "model_family": server_metadata.get("model_family"),
            "model_path": server_metadata.get("model_path"),
            "inference_config": server_metadata.get("inference_config", {}),
        },
    }


def _setup_resume_config(
    *,
    run_dir: Path,
    requested_config: dict[str, Any],
    records: dict[EpisodeKey, dict[str, Any]],
    requested_keys: set[EpisodeKey],
) -> None:
    """Persist a run contract and reject mixed-configuration resumes."""

    unexpected = sorted(set(records) - requested_keys)
    if unexpected:
        raise ValueError(
            "Cannot resume because episodes.jsonl contains entries outside the "
            f"requested run: {unexpected[:10]}"
        )

    prompt_metadata = requested_config["prompts"]
    for key, record in records.items():
        expected_prompt = prompt_metadata[key[1]]
        if record.get("prompt") != expected_prompt:
            raise ValueError(
                "Cannot resume with different prompt configuration for "
                f"repeat/task/episode={key}: existing={record.get('prompt')}, "
                f"requested={expected_prompt}"
            )

    config_path = run_dir / "run_config.json"
    if config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != requested_config:
            raise ValueError(
                "Cannot resume an Oracle Plus run with different settings. "
                f"Existing configuration: {existing_config}; requested: "
                f"{requested_config}. Choose a new RUN_NAME or use overwrite."
            )
        return

    progress_path = run_dir / "progress.json"
    if progress_path.exists() and not records:
        existing_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if existing_progress:
            raise ValueError(
                "Cannot safely resume: progress.json contains results but "
                "episodes.jsonl is missing or empty."
            )
    _write_json(config_path, requested_config)


def _write_vlm_log(path: Path, value: Any) -> None:
    """Write valid JSON with five blank lines between consecutive VLM steps."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    content = json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n"
    content = content.replace(
        '\n    },\n    {\n      "step":',
        '\n    },\n\n\n\n\n\n    {\n      "step":',
    )
    temporary_path.write_text(content, encoding="utf-8")
    os.replace(temporary_path, path)


def _compact_vlm_step(decision_record: dict[str, Any]) -> dict[str, Any]:
    """Keep only the prompt, images, and VLM/controller result for one step."""

    return {
        "step": decision_record["step"],
        "simulator_frame": decision_record["simulator_frame"],
        "messages": decision_record["messages"],
        "images": decision_record["images"],
        "raw_response": decision_record["raw_response"],
        "decision": decision_record["decision"],
        "resolved_action_name": decision_record["resolved_action_name"],
    }


def _simulator_frame(
    info: dict[str, Any],
    *,
    fallback: int,
) -> int:
    """Read the cumulative simulator frame from an action result."""

    value = _jsonable(info.get("elapsed_steps"))
    while isinstance(value, list):
        if not value:
            return fallback
        value = value[-1]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return int(value)


def _repeat_gpu_summary(
    *,
    server_start: dict[str, Any],
    eval_start: dict[str, Any],
    server_interval: dict[str, Any],
    eval_interval: dict[str, Any],
) -> dict[str, Any]:
    """Combine the two Oracle Plus processes into one repeat-level GPU view."""

    gpu_idx = server_start.get("gpu_idx")
    if gpu_idx != eval_start.get("gpu_idx"):
        raise RuntimeError(
            "VLM server and eval process are not using the same physical GPU: "
            f"server={gpu_idx!r}, eval={eval_start.get('gpu_idx')!r}"
        )
    sources = (server_start, eval_start, server_interval, eval_interval)
    errors = [source.get("error") for source in sources if source.get("error")]
    required = (
        server_start.get("global_used_gb"),
        server_start.get("current_process_used_gb"),
        eval_start.get("current_process_used_gb"),
        server_interval.get("average_current_process_gb"),
        eval_interval.get("average_current_process_gb"),
    )
    if errors or any(value is None for value in required):
        result: dict[str, Any] = {"gpu_idx": gpu_idx}
        result["error"] = "; ".join(str(error) for error in errors) or (
            "Incomplete repeat-level GPU measurements"
        )
        return result
    (
        global_used,
        server_used,
        eval_used,
        server_average,
        eval_average,
    ) = (float(value) for value in required)
    return {
        "gpu_idx": gpu_idx,
        "other_processes_before_repeat_gb": round(
            max(global_used - server_used - eval_used, 0.0), 3
        ),
        "current_program_average_during_repeat_gb": round(
            server_average + eval_average, 3
        ),
    }


def _column_values(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _to_vector(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _record_video_observations(
    recorder: RolloutRecorder,
    observations: dict[str, Any],
    *,
    subgoal: str | None,
) -> int:
    fronts = _column_values(observations.get("front_rgb_list", []))
    wrists = _column_values(observations.get("wrist_rgb_list", []))
    joints = _column_values(observations.get("joint_state_list", []))
    grippers = _column_values(observations.get("gripper_state_list", []))
    frame_count = min(len(fronts), len(wrists), len(joints), len(grippers))
    for index in range(frame_count):
        joint_state = _to_vector(joints[index])
        gripper_state = _to_vector(grippers[index])
        state = np.concatenate([joint_state, gripper_state[:1]]).astype(np.float32)
        recorder.record(
            image=_to_rgb(fronts[index]),
            wrist_image=_to_rgb(wrists[index]),
            state=state,
            action=None,
            subgoal=subgoal,
        )
    return frame_count


def _release_dense_observation_batch(observations: dict[str, Any]) -> None:
    """Release dense native-step arrays after retaining request/video frames."""

    for value in observations.values():
        if isinstance(value, list):
            value.clear()


def _filename_text(value: str, *, limit: int = 100) -> str:
    normalized = "_".join(value.strip().split())
    safe = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in normalized
    )
    return safe[:limit] or "unknown_goal"


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
    runtime = OraclePlusPlannerRuntime(
        env,
        task,
        gui_render=False,
        max_retained_steps=args.history_size,
    )
    repeat_dir = run_dir / task / f"repeat_{repeat_id}"
    episode_dir = repeat_dir / f"ep_{episode_id}"
    video_dir = repeat_dir / "videos"
    vlm_log_path = repeat_dir / "vlm_logs" / f"{task}_ep_{episode_id}.json"
    episode_dir.mkdir(parents=True, exist_ok=True)
    recorder: RolloutRecorder | None = None
    record: dict[str, Any] = {
        "task": task,
        "episode_id": episode_id,
        "repeat_id": repeat_id,
        "task_goal": None,
        "task_goal_candidates": [],
        "status": "error",
        "decisions": [],
        "artifacts": {
            "episode_log": str((episode_dir / "episode.json").resolve()),
            "vlm_log": str(vlm_log_path.resolve()),
            "image_dir": str(episode_dir.resolve()),
            "video": None,
        },
        "prompt": get_prompt(
            task,
            include_description=args.include_description,
        ).metadata(),
    }
    try:
        execution = None
        obs, reset_info = runtime.reset()
        goal_candidates = _task_goal_candidates(reset_info)
        goal = _task_goal(reset_info)
        record["task_goal"] = goal
        record["task_goal_candidates"] = goal_candidates
        recorder = RolloutRecorder(video_dir, goal, fps=30, streaming=True)
        _record_video_observations(
            recorder,
            obs,
            subgoal="[initializing...]",
        )
        adapter = build_action_adapter(task, runtime)
        if tuple(adapter.point_actions) != get_point_action_names(task):
            raise RuntimeError(
                f"{task} target-point registries do not match: "
                f"prompt={get_point_action_names(task)}, "
                f"adapter={adapter.point_actions}"
            )
        current_pair = _current_pair(obs)
        image_step = 0
        initial_pair_paths, image_step = _write_observation_steps(
            episode_dir,
            [current_pair],
            start_step=image_step,
        )
        current_pair_paths = initial_pair_paths[0]
        current_image_step = 0
        current_simulator_frame = 0
        initial_paths = list(current_pair_paths)
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
                "step": current_image_step,
                "simulator_frame": current_simulator_frame,
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

            def record_dense_step(step: Any) -> None:
                observations = step[0]
                _record_video_observations(
                    recorder,
                    observations,
                    subgoal=active_subgoal,
                )

            execution = adapter.execute(
                resolved_action_name,
                active_subgoal,
                decision.point,
                step_callback=record_dense_step,
            )
            action_frame_count = execution.simulator_frame_count
            recent_pairs = _recent_pairs(execution.obs_batch, args.history_size)
            _release_dense_observation_batch(execution.obs_batch)
            gpu_monitor.sample()
            last_status = execution.status
            current_simulator_frame = _simulator_frame(
                execution.info,
                fallback=current_simulator_frame + action_frame_count,
            )
            if not recent_pairs:
                recent_pairs = [current_pair]
                recent_pair_paths = [current_pair_paths]
                current_image_step_after = current_image_step
            else:
                recent_pair_paths, image_step = _write_observation_steps(
                    episode_dir,
                    recent_pairs,
                    start_step=image_step,
                )
                current_image_step_after = image_step - 1
            current_pair_after = recent_pairs[-1]
            record["decisions"][-1]["execution"] = {
                "resolved_action_name": resolved_action_name,
                "next_subgoal": active_subgoal,
                "point": decision.point,
                "status": execution.status,
                "stopped": execution.stopped,
                "info": _jsonable(execution.info),
                "simulator_frame_count": action_frame_count,
            }
            if execution.stopped:
                record["status"] = execution.status
                break

            image_paths = list(current_pair_paths)
            for pair_paths in recent_pair_paths:
                image_paths.extend(pair_paths)
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
                    "step": current_image_step_after,
                    "simulator_frame": current_simulator_frame,
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
            current_pair_paths = recent_pair_paths[-1]
            current_image_step = current_image_step_after
        else:
            record["status"] = "decision_limit"
    except Exception as error:
        record["status"] = "error"
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc()
    finally:
        try:
            if recorder is not None:
                record["video_frame_count"] = recorder.frame_count
                record["video_fps"] = recorder.fps
                record["video_duration_seconds"] = (
                    recorder.frame_count / recorder.fps
                )
            if recorder is not None and (
                recorder.has_frames or recorder.stream_error is not None
            ):
                video_filename = (
                    f"{task}_ep_{episode_id}_{record['status']}_"
                    f"{_filename_text(record['task_goal'] or 'unknown')}.mp4"
                )
                video_path = video_dir / video_filename
                try:
                    recorder.save_video(video_filename)
                    record["artifacts"]["video"] = str(video_path.resolve())
                except Exception as video_error:
                    record["video_error"] = (
                        f"{type(video_error).__name__}: {video_error}"
                    )
        finally:
            if execution is not None:
                _release_dense_observation_batch(execution.obs_batch)
            if recorder is not None:
                recorder.total_images.clear()
                recorder.close()
            try:
                runtime.close()
            except Exception as cleanup_error:
                record["cleanup_error"] = (
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            gc.collect()
            record["memory_after_cleanup"] = {
                "eval_process": process_memory_snapshot(),
                "cgroup": cgroup_memory_snapshot(),
            }

    output_path = episode_dir / "episode.json"
    _write_vlm_log(
        vlm_log_path,
        {
            "task": task,
            "episode_id": episode_id,
            "repeat_id": repeat_id,
            "task_goal": record["task_goal"],
            "task_goal_candidates": record["task_goal_candidates"],
            "status": record["status"],
            "prompt": record["prompt"],
            "steps": [
                _compact_vlm_step(decision_record)
                for decision_record in record["decisions"]
            ],
            "error": record.get("error"),
            "traceback": record.get("traceback"),
        },
    )
    _write_json(output_path, record)
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
    resuming = run_dir.exists() and not args.overwrite
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
        print(f"[oracle_plus] Overwriting the evaluation at {run_dir}")
    elif resuming:
        print(f"[oracle_plus] Resuming the evaluation at {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_gpu_monitor = GpuMemoryMonitor(
        os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    )
    model = VLMSubgoalPlannerClient(args.vlm_host, args.vlm_port)
    if model.server_metadata.get("protocol") != "oracle_plus_vlm_v1":
        raise RuntimeError(
            "Connected server does not provide the Oracle Plus VLM protocol: "
            f"{model.server_metadata!r}"
        )

    predictions_path = run_dir / "episodes.jsonl"
    records_by_key = _load_episode_records(predictions_path)
    requested_keys = {
        (repeat_id, task, episode_id)
        for repeat_id in range(1, args.num_repeats + 1)
        for task in tasks
        for episode_id in episode_ids
    }
    requested_config = _requested_run_config(
        args=args,
        tasks=tasks,
        episode_ids=episode_ids,
        server_metadata=model.server_metadata,
    )
    _setup_resume_config(
        run_dir=run_dir,
        requested_config=requested_config,
        records=records_by_key,
        requested_keys=requested_keys,
    )
    completed_keys = {
        key for key, record in records_by_key.items() if _record_is_complete(record)
    }
    if resuming:
        retry_count = len(records_by_key) - len(completed_keys)
        print(
            f"[oracle_plus] Resume found {len(completed_keys)} completed episode "
            f"run(s) and {retry_count} error episode run(s) to retry."
        )

    total_episode_runs = len(requested_keys)
    if (
        len(completed_keys) == total_episode_runs
        and (run_dir / "summary.json").exists()
    ):
        print(
            f"[oracle_plus] Evaluation is already complete at "
            f"{run_dir / 'summary.json'}; leaving existing results unchanged."
        )
        model.close()
        eval_gpu_monitor.close()
        return

    repeat_runtime: dict[int, dict[str, Any]] = {}
    progress_state = _progress_from_records(records_by_key)
    eval_log_path = run_dir / "eval.log"
    terminal_stderr = sys.stderr
    success_count = sum(
        record.get("status") == "success"
        for key, record in records_by_key.items()
        if key in completed_keys
    )
    error_count = 0
    progress = tqdm(
        total=total_episode_runs,
        initial=len(completed_keys),
        desc="oracle_plus",
        unit="episode",
        dynamic_ncols=True,
        leave=True,
        file=terminal_stderr,
        bar_format=(
            "{desc} |{bar}| {n_fmt}/{total_fmt} episodes "
            "({percentage:3.0f}%) [elapsed={elapsed}, remaining={remaining}, "
            "{rate_fmt}] {postfix}"
        ),
    )
    with eval_log_path.open("a", encoding="utf-8") as eval_log:
        for repeat_id in range(1, args.num_repeats + 1):
            pending_repeat_keys = {
                (repeat_id, task, episode_id)
                for task in tasks
                for episode_id in episode_ids
                if (repeat_id, task, episode_id) not in completed_keys
            }
            if not pending_repeat_keys:
                continue
            eval_gpu_sample, eval_gpu_start = eval_gpu_monitor.begin_interval()
            server_gpu_start = model.start_repeat(repeat_id)["gpu_start"]
            repeat_key = f"repeat_{repeat_id}"
            progress_state.setdefault(repeat_key, {})
            for task in tasks:
                progress_state[repeat_key].setdefault(task, {})
                for episode_id in episode_ids:
                    episode_key = (repeat_id, task, episode_id)
                    if episode_key in completed_keys:
                        continue
                    with redirect_stdout(eval_log), redirect_stderr(eval_log):
                        record = run_episode(
                            task=task,
                            episode_id=episode_id,
                            repeat_id=repeat_id,
                            args=args,
                            model=model,
                            gpu_monitor=eval_gpu_monitor,
                            run_dir=run_dir,
                        )
                    records_by_key[episode_key] = record
                    _write_episode_records(predictions_path, records_by_key)
                    eval_log.flush()
                    progress_state[repeat_key][task][str(episode_id)] = (
                        True
                        if record["status"] == "success"
                        else record["status"]
                    )
                    _write_json(run_dir / "progress.json", progress_state)
                    success_count += int(record["status"] == "success")
                    error_count += int(record["status"] == "error")
                    if _record_is_complete(record):
                        completed_keys.add(episode_key)
                    memory = record.get("memory_after_cleanup", {})
                    eval_memory = memory.get("eval_process", {})
                    cgroup_memory = memory.get("cgroup", {})
                    progress.set_postfix(
                        success=success_count,
                        error=error_count,
                        rss_gb=eval_memory.get("rss_gb"),
                        cgroup_gb=cgroup_memory.get("current_gb"),
                        refresh=False,
                    )
                    progress.update(1)
            server_gpu_interval = model.end_repeat(repeat_id)["gpu_interval"]
            eval_gpu_interval = eval_gpu_monitor.end_interval(eval_gpu_sample)
            repeat_runtime[repeat_id] = {
                "gpu": _repeat_gpu_summary(
                    server_start=server_gpu_start,
                    eval_start=eval_gpu_start,
                    server_interval=server_gpu_interval,
                    eval_interval=eval_gpu_interval,
                ),
                "measurement_scope": "current_invocation_resume_segment",
                "measured_episode_runs": len(pending_repeat_keys),
            }
    progress.close()

    records = _ordered_episode_records(records_by_key)
    successes = sum(record["status"] == "success" for record in records)
    vlm_step_times_ms = [
        float(decision["vlm_latency_ms"])
        for record in records
        for decision in record["decisions"]
    ]
    program_step_times_ms = [
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
        repeat_vlm_times_ms = [
            float(decision["vlm_latency_ms"])
            for record in repeat_records
            for decision in record["decisions"]
        ]
        repeat_program_times_ms = [
            float(decision["action_request_ms"])
            for record in repeat_records
            for decision in record["decisions"]
        ]
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
                "gpu_and_timing": {
                    **repeat_runtime.get(
                        repeat_id,
                        {
                            "gpu": {
                                "error": (
                                    "Repeat was completed by an earlier invocation; "
                                    "repeat-level GPU interval is unavailable here."
                                )
                            },
                            "measurement_scope": "previous_invocation",
                            "measured_episode_runs": 0,
                        },
                    ),
                    "vlm_single_step": latency_summary(repeat_vlm_times_ms),
                    "program_single_step": latency_summary(
                        repeat_program_times_ms
                    ),
                },
            }
        )
    vlm_metrics_error = None
    try:
        vlm_metrics = model.metrics()
    except Exception as error:
        vlm_metrics_error = f"{type(error).__name__}: {error}"
        with eval_log_path.open("a", encoding="utf-8") as eval_log:
            eval_log.write("\nVLM metrics request failed:\n")
            eval_log.write(traceback.format_exc())
            eval_log.flush()
        metadata = model.server_metadata
        vlm_metrics = {
            "protocol": metadata.get("protocol", "oracle_plus_vlm_v1"),
            "model_family": metadata.get("model_family", "Qwen3.8-27B"),
            "model_path": metadata.get("model_path", "unknown"),
            "inference_config": metadata.get("inference_config", {}),
            "gpu": {"error": vlm_metrics_error},
            "model_load_seconds": metadata.get("model_load_seconds"),
        }
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
        "vlm_server_metrics_error": vlm_metrics_error,
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
            "vlm_single_step": latency_summary(vlm_step_times_ms),
            "program_single_step": latency_summary(program_step_times_ms),
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "log.json", summary)
    model.close()
    eval_gpu_monitor.close()


if __name__ == "__main__":
    main()
