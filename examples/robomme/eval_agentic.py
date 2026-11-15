"""End-to-end Agentic evaluation: init plan, VLM loop, grounding, actions."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import string
import sys
import traceback
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import is_dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party" / "robomme_benchmark" / "src"
ROBOMME_EXAMPLES_ROOT = REPO_ROOT / "examples" / "robomme"
for root in (REPO_ROOT / "src", THIRD_PARTY_ROOT, ROBOMME_EXAMPLES_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from harness.actions import ActionExecutor
from harness.actions import build_action_registry
from harness.actions import resolve_action_arguments
from harness.grounding import F2GClient
from harness.grounding import GroundingContractRegistry
from harness.grounding import GroundingPipeline
from harness.grounding import camera_frame_from_robomme
from harness.grounding import depth_statistics
from harness.init_planning import ACTION_CATALOG_PATH
from harness.init_planning import build_init_planning_messages
from harness.init_planning import validate_task_config
from harness.task_memory import TaskMemory
from harness.vlm_client import AgenticVLMClient
from oracle_plus.action_runtime import OraclePlusPlannerRuntime
from robomme.env_record_wrapper import BenchmarkEnvBuilder
from utils import RolloutRecorder


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _episode_ids(value: str) -> list[int] | None:
    if not value.strip():
        return None
    values = [int(item) for item in _csv(value)]
    if any(item < 0 for item in values) or len(values) != len(set(values)):
        raise ValueError("episode IDs must be unique non-negative integers")
    return values


def _safe_name(value: str) -> str:
    value = value.strip()
    if not value or value in {".", ".."} or Path(value).name != value:
        raise argparse.ArgumentTypeError("run name must be one directory name")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-host", default="127.0.0.1")
    parser.add_argument("--vlm-port", type=int, required=True)
    parser.add_argument("--f2g-url", default="http://127.0.0.1:8765")
    parser.add_argument(
        "--pre-eval-gpu-snapshot",
        type=Path,
        required=True,
        help="GPU snapshot captured before the F2G and VLM services start",
    )
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--tasks", default="BinFill")
    parser.add_argument("--episode-ids", default="")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--max-decisions", type=int, default=50)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--history-stride", type=int, default=16)
    parser.add_argument("--max-initial-observations", type=int, default=16)
    parser.add_argument("--planner-attempts", type=int, default=3)
    parser.add_argument("--save-dir", type=Path, default=Path("runs/evaluation/agentic"))
    parser.add_argument("--run-name", type=_safe_name, default="agentic.1")
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--use-task-config-examples",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _to_rgb(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    image = np.asarray(value)
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 4:
        image = image[-1]
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"expected HWC RGB image, got {image.shape}")
    image = image[..., :3]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.max() <= 1:
            image = image * 255
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _columns(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _latest_observation(obs: Mapping[str, Any]) -> dict[str, Any]:
    latest = {}
    for key, value in obs.items():
        if isinstance(value, (list, tuple)):
            if value:
                latest[key] = value[-1]
        else:
            latest[key] = value
    return latest


def _release_columnar_observation(obs: Mapping[str, Any]) -> None:
    for value in obs.values():
        if isinstance(value, list):
            value.clear()


def _observation_pairs(obs: Mapping[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    fronts = _columns(obs["front_rgb_list"])
    wrists = _columns(obs["wrist_rgb_list"])
    count = min(len(fronts), len(wrists))
    if count < 1:
        raise ValueError("observation has no aligned front/wrist images")
    return [(_to_rgb(fronts[index]), _to_rgb(wrists[index])) for index in range(count)]


def _to_vector(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _observation_frame_count(obs: Mapping[str, Any]) -> int:
    """Return the number of complete front/wrist/state observation rows."""

    return min(
        len(_columns(obs.get("front_rgb_list", []))),
        len(_columns(obs.get("wrist_rgb_list", []))),
        len(_columns(obs.get("joint_state_list", []))),
        len(_columns(obs.get("gripper_state_list", []))),
    )


def _record_video_observations(
    recorder: RolloutRecorder,
    obs: Mapping[str, Any],
    *,
    subgoal: str | None,
    point_yx: tuple[int, int] | None = None,
    start_index: int = 0,
    end_index: int | None = None,
    is_video_demo: bool = False,
) -> int:
    fronts = _columns(obs.get("front_rgb_list", []))
    wrists = _columns(obs.get("wrist_rgb_list", []))
    joints = _columns(obs.get("joint_state_list", []))
    grippers = _columns(obs.get("gripper_state_list", []))
    frame_count = min(len(fronts), len(wrists), len(joints), len(grippers))
    start = min(max(int(start_index), 0), frame_count)
    end = (
        frame_count
        if end_index is None
        else min(max(int(end_index), start), frame_count)
    )
    for index in range(start, end):
        state = np.concatenate(
            [_to_vector(joints[index]), _to_vector(grippers[index])[:1]]
        ).astype(np.float32)
        recorder.record(
            image=_to_rgb(fronts[index]),
            wrist_image=_to_rgb(wrists[index]),
            state=state,
            action=None,
            is_video_demo=is_video_demo,
            subgoal=subgoal,
            point_yx=point_yx,
        )
    return end - start


def _record_reset_demonstration(
    recorder: RolloutRecorder,
    obs: Mapping[str, Any],
) -> int:
    """Record reset demonstration frames, excluding the execution initial state."""

    frame_count = _observation_frame_count(obs)
    if frame_count < 2:
        return 0
    return _record_video_observations(
        recorder,
        obs,
        subgoal="[demonstration]",
        end_index=frame_count - 1,
        is_video_demo=True,
    )


def _grounding_point_yx(grounding: Mapping[str, Any]) -> tuple[int, int] | None:
    targets = grounding.get("targets", {})
    if not isinstance(targets, Mapping):
        return None
    for target in targets.values():
        if not isinstance(target, Mapping):
            continue
        point = target.get("point_xy")
        if (
            isinstance(point, (list, tuple))
            and len(point) == 2
            and all(isinstance(value, (int, float)) for value in point)
        ):
            return int(round(point[1])), int(round(point[0]))
    return None


def _filename_text(value: str, *, limit: int = 100) -> str:
    normalized = "_".join(value.strip().split())
    safe = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in normalized
    )
    return safe[:limit] or "unknown_goal"


def _sample_pairs(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    maximum: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(pairs) <= maximum:
        return pairs
    indices = np.linspace(0, len(pairs) - 1, maximum, dtype=int)
    return [pairs[int(index)] for index in indices]


def _task_goal(info: Mapping[str, Any]) -> str:
    value = info.get("task_goal")
    while isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("environment task_goal is empty")
        value = value[0]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"environment task_goal is invalid: {value!r}")
    return value.strip()


def _save_observation(
    directory: Path,
    *,
    observation_id: str,
    pair: tuple[np.ndarray, np.ndarray],
    include_wrist: bool,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    overall_path = (directory / f"{observation_id}_overall.png").resolve()
    iio.imwrite(overall_path, pair[0])
    result: dict[str, Any] = {
        "observation_id": observation_id,
        "overall": {
            "image": "<image>",
            "image_path": str(overall_path),
            "image_ref": observation_id,
        },
    }
    if include_wrist:
        wrist_path = (directory / f"{observation_id}_wrist.png").resolve()
        iio.imwrite(wrist_path, pair[1])
        result["wrist"] = {
            "image": "<image>",
            "image_path": str(wrist_path),
            "image_ref": f"{observation_id}_wrist",
        }
    return result


def _save_observation_window(
    directory: Path,
    *,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    prefix: str,
    include_wrist: bool,
    current_last: bool,
) -> list[dict[str, Any]]:
    result = []
    for index, pair in enumerate(pairs):
        observation_id = (
            "current_front"
            if current_last and index == len(pairs) - 1
            else f"{prefix}_{index}_front"
        )
        result.append(
            _save_observation(
                directory,
                observation_id=observation_id,
                pair=pair,
                include_wrist=include_wrist,
            )
        )
    return result


def _collect_images(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        if value.get("image") == "<image>":
            image_path = value.get("image_path")
            if not isinstance(image_path, str):
                raise ValueError("an <image> object is missing image_path")
            paths.append(image_path)
        for key, child in value.items():
            if key not in {"image", "image_path"}:
                paths.extend(_collect_images(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            paths.extend(_collect_images(child))
    return paths


def _render_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(_jsonable(value), ensure_ascii=False, indent=2)


def _build_planner_messages(
    task_config: Mapping[str, Any],
    *,
    prompt_name: str,
    inputs: Mapping[str, Any],
) -> tuple[list[dict[str, str]], list[str]]:
    prompt = task_config["vlm_prompts"][prompt_name]
    template = prompt["user_prompt_template"]
    rendered: dict[str, str] = {}
    image_paths: list[str] = []
    formatter = string.Formatter()
    for _, field_name, _, _ in formatter.parse(template):
        if field_name is None:
            continue
        if field_name not in inputs:
            raise ValueError(f"planner input is missing {field_name!r}")
        rendered[field_name] = _render_value(inputs[field_name])
        image_paths.extend(_collect_images(inputs[field_name]))
    user_prompt = template.format_map(rendered)
    placeholder_count = user_prompt.count("<image>")
    if placeholder_count != len(image_paths):
        raise ValueError(
            f"planner prompt has {placeholder_count} image placeholders but "
            f"{len(image_paths)} paths"
        )
    return (
        [
            {"role": "system", "content": prompt["system_prompt"]},
            {"role": "user", "content": user_prompt},
        ],
        image_paths,
    )


def _active_keyframe_policy(memory: TaskMemory) -> int:
    if not memory.uses_keyframes:
        return 0
    state = memory.load_state()
    for index, policy in enumerate(memory.keyframe_policies):
        value: Any = state if state else {}
        for part in str(policy["task_state_field"]).split("."):
            value = value.get(part) if isinstance(value, Mapping) else None
        if value is None:
            return index
        if policy["capture_mode"] == "sequence" and len(value) < policy["frame_count"]:
            return index
    return len(memory.keyframe_policies) - 1


def _observation_path_map(values: list[dict[str, Any]]) -> dict[str, str]:
    return {
        item["observation_id"]: item["overall"]["image_path"]
        for item in values
    }


def _validate_next_grounding(
    memory: TaskMemory,
    contracts: GroundingContractRegistry,
    output: Mapping[str, Any],
    image_shape: tuple[int, int],
) -> None:
    key = output.get("next_subgoal_key")
    if key == "STOP":
        normalized = output.get("grounding")
        if normalized != {
            "status": "not_required",
            "image_ref": None,
            "targets": {},
        }:
            raise ValueError("STOP requires grounding.status=not_required")
        return
    resolved = memory.resolver.resolve(
        subgoal_key=key,
        subgoal=output.get("next_subgoal"),
        arguments=output.get("arguments", {}),
    )
    grounding = contracts.validate(
        resolved.action_name,
        output.get("grounding"),
        image_shape=image_shape,
    )
    if grounding["status"] == "ambiguous":
        raise ValueError("grounding is ambiguous; select a visible target")
    if grounding["status"] == "ok" and grounding["image_ref"] != "current_front":
        raise ValueError("grounding.image_ref must be current_front")


def _infer_and_commit(
    *,
    client: AgenticVLMClient,
    messages: list[dict[str, str]],
    images: list[str],
    attempts: int,
    planning_phase: str,
    validate_and_commit: Any,
    attempt_callback: Any | None = None,
) -> tuple[dict[str, Any], str, float, float]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request_messages = [dict(message) for message in messages]
        if last_error is not None:
            request_messages[-1]["content"] += (
                "\n\nYour previous JSON was rejected by the trusted harness: "
                f"{type(last_error).__name__}: {last_error}. Return a corrected "
                "complete JSON object only."
            )
        output, raw, latency, request_latency = client.infer_json(
            messages=request_messages,
            images=images,
            planning_phase=planning_phase,
        )
        try:
            validate_and_commit(output)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            if attempt_callback is not None:
                attempt_callback(
                    {
                        "attempt": attempt,
                        "messages": request_messages,
                        "images": images,
                        "output": output,
                        "raw_response": raw,
                        "vlm_latency_ms": latency,
                        "request_to_json_ms": request_latency,
                        "validation_error": f"{type(error).__name__}: {error}",
                    }
                )
            last_error = error
            if attempt == attempts:
                raise
        else:
            if attempt_callback is not None:
                attempt_callback(
                    {
                        "attempt": attempt,
                        "messages": request_messages,
                        "images": images,
                        "output": output,
                        "raw_response": raw,
                        "vlm_latency_ms": latency,
                        "request_to_json_ms": request_latency,
                        "validation_error": None,
                    }
                )
            return output, raw, latency, request_latency
    raise AssertionError("planner retry loop exited unexpectedly")


def _decision_summary(
    memory: TaskMemory,
    output: Mapping[str, Any],
) -> dict[str, Any]:
    key = output.get("next_subgoal_key")
    grounding = _jsonable(output.get("grounding"))
    if key == "STOP":
        return {
            "next_subgoal_key": "STOP",
            "next_subgoal": "STOP",
            "action_name": None,
            "arguments": {},
            "grounding": grounding,
            "predicted_points_xy": {},
        }
    resolved = memory.resolver.resolve(
        subgoal_key=key,
        subgoal=output.get("next_subgoal"),
        arguments=output.get("arguments", {}),
    )
    targets = grounding.get("targets", {}) if isinstance(grounding, dict) else {}
    points = {
        role: target.get("point_xy")
        for role, target in targets.items()
        if isinstance(target, Mapping)
    }
    return {
        "next_subgoal_key": resolved.subgoal_key,
        "next_subgoal": resolved.subgoal,
        "action_name": resolved.action_name,
        "arguments": resolved.arguments,
        "grounding": grounding,
        "predicted_points_xy": points,
    }


def _write_information_source_artifacts(
    memory: TaskMemory,
    directory: Path,
) -> None:
    state = memory.load_state()
    if not state:
        return
    directory.mkdir(parents=True, exist_ok=True)
    if memory.uses_video_subgoal_list:
        sequence = [
            {
                "subgoal_key": item["subgoal_key"],
                "subgoal": item["subgoal"],
                "arguments": item.get("arguments", {}),
            }
            for item in state["plan"]["sequence"]
        ]
        _write_json(directory / "video_subgoal_list.json", sequence)
    if memory.uses_keyframes:
        _write_json(
            directory / "keyframes.json",
            {
                "policies": memory.keyframe_policies,
                "persistent_facts": state.get("persistent_facts", {}),
            },
        )


def _describe_keyframe(
    *,
    client: AgenticVLMClient,
    policy: Mapping[str, Any],
    observation: Mapping[str, Any],
    attempts: int,
) -> str:
    instruction = policy.get("description_instruction")
    messages = [
        {
            "role": "system",
            "content": (
                "Describe one saved visual observation. Return JSON only with "
                "exactly one non-empty field named description. Follow the "
                "user's evidence instruction and do not infer hidden state."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Evidence instruction: {instruction}\n\n"
                "Observation:\n"
                f"{_render_value(observation)}"
            ),
        },
    ]
    images = _collect_images(observation)
    description: str | None = None

    def validate(output: dict[str, Any]) -> None:
        nonlocal description
        if set(output) != {"description"}:
            raise ValueError("keyframe description output has extra fields")
        value = output["description"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("keyframe description must be non-empty")
        description = value.strip()

    _infer_and_commit(
        client=client,
        messages=messages,
        images=images,
        attempts=attempts,
        planning_phase="auxiliary",
        validate_and_commit=validate,
    )
    if description is None:
        raise AssertionError("keyframe description was not produced")
    return description


def _generate_task_config(
    *,
    task: str,
    output_path: Path,
    client: AgenticVLMClient,
    use_examples: bool,
    attempts: int,
    verification_k: int,
    verification_interval: int,
) -> dict[str, Any]:
    messages, catalog = build_init_planning_messages(
        task_name=task,
        output_path=output_path,
        use_examples=use_examples,
        verification_k=verification_k,
        verification_interval=verification_interval,
    )
    config: dict[str, Any] | None = None

    def validate(value: dict[str, Any]) -> None:
        nonlocal config
        config = validate_task_config(value, catalog, task_name=task)

    _infer_and_commit(
        client=client,
        messages=messages,
        images=[],
        attempts=attempts,
        planning_phase="init_planning",
        validate_and_commit=validate,
    )
    if config is None:
        raise AssertionError("init planning returned no validated task config")
    _write_json(output_path, config)
    return config


def _flag(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return bool(np.asarray(value).any())


def _save_masks(directory: Path, grounded: Any) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for role, segmentation in grounded.masks.items():
        path = (directory / f"{role}.png").resolve()
        iio.imwrite(path, segmentation.mask.astype(np.uint8) * 255)
        paths[role] = str(path)
    return paths


def _save_grounding_input(directory: Path, frame: Any) -> Path:
    """Persist the aligned RGB-D camera inputs needed to replay F2G."""

    directory.mkdir(parents=True, exist_ok=True)
    path = (directory / "grounding_input.npz").resolve()
    np.savez_compressed(
        path,
        rgb=frame.rgb,
        depth=frame.depth,
        intrinsic=frame.intrinsic,
        world_to_camera=frame.world_to_camera,
        depth_scale=np.asarray(frame.depth_scale, dtype=np.float64),
    )
    return path


def _save_grounding_artifact(
    directory: Path,
    *,
    execution: Mapping[str, Any],
    grounded: Any,
) -> dict[str, str]:
    mask_paths = _save_masks(directory, grounded)
    _write_json(
        directory / "grounding.json",
        {
            "action_name": execution["action_name"],
            "subgoal_key": execution["subgoal_key"],
            "subgoal": execution["subgoal"],
            "arguments": execution["arguments"],
            "vlm_grounding": execution["grounding"],
            "action_request": grounded.request,
            "masks": {
                role: {
                    "path": mask_paths[role],
                    "profile": segmentation.profile,
                    "confidence": segmentation.confidence,
                    "diagnostics": segmentation.diagnostics,
                }
                for role, segmentation in grounded.masks.items()
            },
        },
    )
    return mask_paths


def run_episode(
    *,
    task: str,
    episode_id: int,
    repeat_id: int,
    args: argparse.Namespace,
    task_config: dict[str, Any],
    client: AgenticVLMClient,
    run_dir: Path,
) -> dict[str, Any]:
    repeat_dir = run_dir / task / f"repeat_{repeat_id}"
    episode_dir = repeat_dir / "images" / f"ep_{episode_id}"
    memory_dir = repeat_dir / "task_memory" / f"ep_{episode_id}"
    grounding_dir = repeat_dir / "grounding" / f"ep_{episode_id}"
    source_dir = repeat_dir / "information_sources" / f"ep_{episode_id}"
    video_episode_dir = repeat_dir / "videos" / f"ep_{episode_id}"
    example_dir = repeat_dir / "example"
    vlm_log_path = repeat_dir / "vlm_logs" / f"{task}_ep_{episode_id}.json"
    for stale_dir in (
        episode_dir,
        memory_dir,
        grounding_dir,
        source_dir,
        video_episode_dir,
    ):
        if stale_dir.exists():
            shutil.rmtree(stale_dir)
    if vlm_log_path.exists():
        vlm_log_path.unlink()
    episode_dir.mkdir(parents=True)
    vlm_trace: dict[str, Any] = {
        "task": task,
        "episode_id": episode_id,
        "repeat_id": repeat_id,
        "calls": [],
    }
    record: dict[str, Any] = {
        "task": task,
        "episode_id": episode_id,
        "repeat_id": repeat_id,
        "status": "error",
        "task_goal": None,
        "decisions": [],
        "artifacts": {
            "episode_log": str((video_episode_dir / "episode.json").resolve()),
            "vlm_log": str(vlm_log_path.resolve()),
            "image_dir": str(episode_dir.resolve()),
            "task_memory_dir": str(memory_dir.resolve()),
            "grounding_dir": str(grounding_dir.resolve()),
            "information_sources_dir": str(source_dir.resolve()),
            "video": None,
            "example": None,
        },
    }
    runtime: OraclePlusPlannerRuntime | None = None
    recorder: RolloutRecorder | None = None
    example_recorder: RolloutRecorder | None = None
    try:
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
            include_front_depth=True,
            include_front_camera_extrinsic=True,
            include_front_camera_intrinsic=True,
        )
        verification_k = int(task_config["verification"]["k"])
        verification_stride = int(task_config["verification"]["interval"])
        retained = verification_k * verification_stride + 1
        runtime = OraclePlusPlannerRuntime(
            env,
            task,
            gui_render=False,
            max_retained_steps=retained,
        )
        obs, info = runtime.reset()
        task_goal = _task_goal(info)
        record["task_goal"] = task_goal
        recorder = RolloutRecorder(
            video_episode_dir,
            task_goal,
            fps=30,
            streaming=True,
        )
        reset_frame_count = _observation_frame_count(obs)
        if reset_frame_count:
            _record_video_observations(
                recorder,
                obs,
                subgoal="[initializing...]",
                start_index=reset_frame_count - 1,
            )
        example_recorder = RolloutRecorder(
            example_dir,
            task_goal,
            fps=30,
            streaming=True,
        )
        example_frame_count = _record_reset_demonstration(example_recorder, obs)
        record["example_video_frame_count"] = example_frame_count
        record["example_video_fps"] = 30
        if example_frame_count:
            example_filename = (
                f"{task}_ep_{episode_id}_example_"
                f"{_filename_text(task_goal)}.mp4"
            )
            example_path = example_dir / example_filename
            try:
                example_recorder.save_video(example_filename)
                record["artifacts"]["example"] = str(example_path.resolve())
            except Exception as example_video_error:
                record["example_video_error"] = (
                    f"{type(example_video_error).__name__}: "
                    f"{example_video_error}"
                )
        registry = build_action_registry()
        executor = ActionExecutor(runtime, registry)
        contracts = GroundingContractRegistry.from_catalog(ACTION_CATALOG_PATH)
        pipeline = GroundingPipeline(
            f2g_client=F2GClient(base_url=args.f2g_url)
        )
        memory = TaskMemory(
            memory_dir,
            task_config,
            set(registry),
        )
        views = set(task_config["verification"].get("views", ["overall"]))
        include_wrist = "wrist" in views

        initial_pairs = _sample_pairs(
            _observation_pairs(obs), args.max_initial_observations
        )
        call_dir = episode_dir / "call_0"
        initial_observations = _save_observation_window(
            call_dir,
            pairs=initial_pairs,
            prefix="initial",
            include_wrist=include_wrist,
            current_last=True,
        )
        policy_index = _active_keyframe_policy(memory)
        first_inputs = memory.build_first_planning_inputs(
            task_goal=task_goal,
            current_observations=initial_observations,
            keyframe_policy_index=policy_index,
        )
        messages, images = _build_planner_messages(
            task_config,
            prompt_name="first_planning",
            inputs=first_inputs,
        )
        image_shape = initial_pairs[-1][0].shape[:2]

        def commit_first(output: dict[str, Any]) -> None:
            _validate_next_grounding(memory, contracts, output, image_shape)
            memory.commit_first_plan(task_goal=task_goal, planner_output=output)

        first_attempts: list[dict[str, Any]] = []
        first_call_log: dict[str, Any] = {
            "call_index": 0,
            "call_type": "first_planning",
            "planner_inputs": first_inputs,
            "messages": messages,
            "images": images,
            "attempts": first_attempts,
            "output": None,
            "raw_response": None,
            "decision": None,
        }
        vlm_trace["calls"].append(first_call_log)
        _write_json(vlm_log_path, vlm_trace)

        def record_first_attempt(attempt: dict[str, Any]) -> None:
            first_attempts.append(attempt)
            _write_json(vlm_log_path, vlm_trace)

        first_output, raw, latency, request_latency = _infer_and_commit(
            client=client,
            messages=messages,
            images=images,
            attempts=args.planner_attempts,
            planning_phase="in_task_planning",
            validate_and_commit=commit_first,
            attempt_callback=record_first_attempt,
        )
        if memory.uses_keyframes:
            selected_id = first_output.get("keyframe_save")
            selected = next(
                (
                    item
                    for item in initial_observations
                    if item["observation_id"] == selected_id
                ),
                None,
            )
            description = (
                _describe_keyframe(
                    client=client,
                    policy=memory.keyframe_policies[policy_index],
                    observation=selected,
                    attempts=args.planner_attempts,
                )
                if selected is not None
                else None
            )
            memory.save_keyframe_selection(
                keyframe_save=selected_id,
                observation_paths=_observation_path_map(initial_observations),
                description=description,
                policy_index=policy_index,
            )
        _write_information_source_artifacts(memory, source_dir)
        first_resolution = _decision_summary(memory, first_output)
        first_call_log.update(
            {
                "output": first_output,
                "raw_response": raw,
                "decision": first_resolution,
                "vlm_latency_ms": latency,
                "request_to_json_ms": request_latency,
            }
        )
        _write_json(vlm_log_path, vlm_trace)
        record["decisions"].append(
            {
                "call": 0,
                "type": "first_planning",
                "output": first_output,
                "raw_response": raw,
                "decision": first_resolution,
                "vlm_latency_ms": latency,
                "request_to_json_ms": request_latency,
            }
        )

        current_obs: Mapping[str, Any] = _latest_observation(obs)
        current_info: Mapping[str, Any] = info
        current_pair = initial_pairs[-1]
        del initial_pairs
        _release_columnar_observation(obs)
        last_action_status = "ongoing"
        for decision_index in range(1, args.max_decisions + 1):
            execution = memory.current_execution()
            if execution is None:
                break
            point_yx = _grounding_point_yx(execution["grounding"])
            _record_video_observations(
                recorder,
                current_obs,
                subgoal=execution["subgoal"],
                point_yx=point_yx,
            )
            frame = camera_frame_from_robomme(
                current_obs,
                current_info,
                depth_scale=args.depth_scale,
                image_ref="current_front",
            )
            if "depth_statistics" not in record:
                record["depth_statistics"] = depth_statistics(frame)
            active_call_index = decision_index - 1
            grounding_call_dir = grounding_dir / f"call_{active_call_index}"
            _save_grounding_input(grounding_call_dir, frame)
            grounded = pipeline.build_action_request(
                action_name=execution["action_name"],
                grounding=execution["grounding"],
                frame=frame,
                arguments=resolve_action_arguments(
                    execution["action_name"],
                    subgoal=execution["subgoal"],
                    arguments=execution["arguments"],
                ),
            )
            mask_paths = _save_grounding_artifact(
                grounding_call_dir,
                execution=execution,
                grounded=grounded,
            )
            memory.record_action_started(mask_paths=mask_paths)
            recent_pairs: deque[tuple[np.ndarray, np.ndarray]] = deque(
                maxlen=verification_k
            )
            native_step = 0

            def collect_step(step: Any) -> None:
                nonlocal native_step
                native_step += 1
                _record_video_observations(
                    recorder,
                    step[0],
                    subgoal=execution["subgoal"],
                    point_yx=point_yx,
                )
                if native_step % verification_stride == 0:
                    recent_pairs.append(_observation_pairs(step[0])[-1])

            action_result = executor.execute(
                grounded.request,
                step_callback=collect_step,
            )
            native_execution = executor.last_execution
            if native_execution is None:
                raise RuntimeError("ActionExecutor did not retain native execution")
            final_pair = _observation_pairs(native_execution.obs_batch)[-1]
            recent_pairs.append(final_pair)
            last_action_status = action_result.status
            memory.record_action_finished(
                runtime_status=action_result.status,
                artifact_refs=list(mask_paths.values()),
                simulator_frame_count=action_result.simulator_frame_count,
            )
            current_obs = _latest_observation(native_execution.obs_batch)
            current_info = native_execution.info
            _release_columnar_observation(native_execution.obs_batch)

            call_dir = episode_dir / f"call_{decision_index}"
            start_observation = _save_observation(
                call_dir,
                observation_id="subgoal_start_front",
                pair=current_pair,
                include_wrist=include_wrist,
            )
            recent_observations = _save_observation_window(
                call_dir,
                pairs=list(recent_pairs),
                prefix="recent",
                include_wrist=include_wrist,
                current_last=True,
            )
            policy_index = _active_keyframe_policy(memory)
            follow_inputs = memory.build_followup_inputs(
                verification_parameters=task_config["verification"],
                subgoal_start_observations=[start_observation],
                recent_observations=recent_observations,
                current_observation=recent_observations[-1],
                keyframe_policy_index=policy_index,
            )
            messages, images = _build_planner_messages(
                task_config,
                prompt_name="followup_planning",
                inputs=follow_inputs,
            )

            def commit_followup(output: dict[str, Any]) -> None:
                _validate_next_grounding(
                    memory,
                    contracts,
                    output,
                    final_pair[0].shape[:2],
                )
                memory.commit_followup(output)

            followup_attempts: list[dict[str, Any]] = []
            followup_call_log: dict[str, Any] = {
                "call_index": decision_index,
                "call_type": "followup_planning",
                "planner_inputs": follow_inputs,
                "messages": messages,
                "images": images,
                "attempts": followup_attempts,
                "output": None,
                "raw_response": None,
                "decision": None,
                "executed_before_call": execution,
                "action_result": _jsonable(action_result),
            }
            vlm_trace["calls"].append(followup_call_log)
            _write_json(vlm_log_path, vlm_trace)

            def record_followup_attempt(attempt: dict[str, Any]) -> None:
                followup_attempts.append(attempt)
                _write_json(vlm_log_path, vlm_trace)

            output, raw, latency, request_latency = _infer_and_commit(
                client=client,
                messages=messages,
                images=images,
                attempts=args.planner_attempts,
                planning_phase="in_task_planning",
                validate_and_commit=commit_followup,
                attempt_callback=record_followup_attempt,
            )
            if memory.uses_keyframes:
                candidates = [start_observation, *recent_observations]
                memory.save_keyframe_selection(
                    keyframe_save=output.get("keyframe_save"),
                    observation_paths=_observation_path_map(candidates),
                    description=(
                        output.get("description")
                        if output.get("keyframe_save") is not None
                        else None
                    ),
                    policy_index=policy_index,
                )
            _write_information_source_artifacts(memory, source_dir)
            resolution = _decision_summary(memory, output)
            followup_call_log.update(
                {
                    "output": output,
                    "raw_response": raw,
                    "decision": resolution,
                    "vlm_latency_ms": latency,
                    "request_to_json_ms": request_latency,
                }
            )
            _write_json(vlm_log_path, vlm_trace)
            record["decisions"].append(
                {
                    "call": decision_index,
                    "type": "followup_planning",
                    "executed": execution,
                    "action_result": _jsonable(action_result),
                    "output": output,
                    "raw_response": raw,
                    "decision": resolution,
                    "vlm_latency_ms": latency,
                    "request_to_json_ms": request_latency,
                    "mask_paths": mask_paths,
                }
            )
            current_pair = final_pair
            if memory.current_execution() is None:
                break
        else:
            raise RuntimeError("maximum planner decisions reached before STOP")

        evaluation = runtime.env.unwrapped.evaluate(solve_complete_eval=True)
        record["evaluation"] = _jsonable(evaluation)
        if _flag(evaluation.get("success", False)):
            record["status"] = "success"
        elif _flag(evaluation.get("fail", False)):
            record["status"] = "fail"
        elif memory.current_execution() is None:
            record["status"] = "completed_without_env_success"
        else:
            record["status"] = last_action_status
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
                video_path = video_episode_dir / video_filename
                try:
                    recorder.save_video(video_filename)
                    record["artifacts"]["video"] = str(video_path.resolve())
                except Exception as video_error:
                    record["video_error"] = (
                        f"{type(video_error).__name__}: {video_error}"
                    )
        finally:
            if recorder is not None:
                recorder.total_images.clear()
                recorder.close()
            if example_recorder is not None:
                example_recorder.total_images.clear()
                example_recorder.close()
            if runtime is not None:
                try:
                    runtime.close()
                except Exception as cleanup_error:
                    record["cleanup_error"] = (
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            gc.collect()
    _write_json(video_episode_dir / "episode.json", record)
    if vlm_trace["calls"]:
        _write_json(vlm_log_path, vlm_trace)
    return record


def main() -> None:
    args = parse_args()
    tasks = _csv(args.tasks)
    episode_ids = _episode_ids(args.episode_ids) or list(range(args.num_episodes))
    for name, value in (
        ("num_repeats", args.num_repeats),
        ("max_decisions", args.max_decisions),
        ("history_size", args.history_size),
        ("history_stride", args.history_stride),
        ("planner_attempts", args.planner_attempts),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if not tasks:
        raise ValueError("--tasks must contain at least one task")

    run_dir = (args.save_dir / args.run_name).expanduser().resolve()
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    pre_eval_gpu = json.loads(
        args.pre_eval_gpu_snapshot.read_text(encoding="utf-8")
    )
    _write_json(run_dir / "pre_eval_gpu.json", pre_eval_gpu)
    client = AgenticVLMClient(args.vlm_host, args.vlm_port)
    if client.server_metadata.get("protocol") != "agentic_vlm_json_v1":
        raise RuntimeError(f"wrong VLM protocol: {client.server_metadata}")
    f2g_health = F2GClient(base_url=args.f2g_url).health()
    requested_config = {
        "tasks": tasks,
        "episode_ids": episode_ids,
        "num_repeats": args.num_repeats,
        "max_steps": args.max_steps,
        "max_decisions": args.max_decisions,
        "history_size": args.history_size,
        "history_stride": args.history_stride,
        "depth_scale": args.depth_scale,
        "use_task_config_examples": args.use_task_config_examples,
        "vlm_model_path": client.server_metadata.get("model_path"),
        "f2g_checkpoint": f2g_health.get("checkpoint"),
    }
    run_config_path = run_dir / "run_config.json"
    if run_config_path.exists() and not args.overwrite:
        existing_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing_config != requested_config:
            raise RuntimeError(
                "resume configuration differs from run_config.json; choose a "
                "new RUN_NAME or set OVERWRITE=true"
            )
    else:
        _write_json(run_config_path, requested_config)

    task_configs: dict[str, dict[str, Any]] = {}
    for task in tasks:
        path = run_dir / "task_configs" / f"{task}.json"
        if path.exists() and not args.overwrite:
            config = json.loads(path.read_text(encoding="utf-8"))
            catalog = json.loads(ACTION_CATALOG_PATH.read_text(encoding="utf-8"))
            task_configs[task] = validate_task_config(
                config,
                catalog,
                task_name=task,
            )
        else:
            print(f"[agentic] init planning: {task}")
            task_configs[task] = _generate_task_config(
                task=task,
                output_path=path,
                client=client,
                use_examples=args.use_task_config_examples,
                attempts=args.planner_attempts,
                verification_k=args.history_size,
                verification_interval=args.history_stride,
            )

    records_path = run_dir / "episodes.jsonl"
    completed: set[tuple[int, str, int]] = set()
    records: list[dict[str, Any]] = []
    if records_path.exists() and not args.overwrite:
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(record)
            if record.get("status") != "error":
                completed.add(
                    (record["repeat_id"], record["task"], record["episode_id"])
                )
    requested = [
        (repeat_id, task, episode_id)
        for repeat_id in range(1, args.num_repeats + 1)
        for task in tasks
        for episode_id in episode_ids
    ]
    progress = tqdm(total=len(requested), initial=len(completed), desc="agentic")
    for repeat_id, task, episode_id in requested:
        key = (repeat_id, task, episode_id)
        if key in completed:
            continue
        record = run_episode(
            task=task,
            episode_id=episode_id,
            repeat_id=repeat_id,
            args=args,
            task_config=task_configs[task],
            client=client,
            run_dir=run_dir,
        )
        records = [
            item
            for item in records
            if (item["repeat_id"], item["task"], item["episode_id"]) != key
        ]
        records.append(record)
        records.sort(key=lambda item: (item["repeat_id"], item["task"], item["episode_id"]))
        records_path.write_text(
            "".join(
                json.dumps(_jsonable(item), ensure_ascii=False) + "\n"
                for item in records
            ),
            encoding="utf-8",
        )
        progress.update(1)
        progress.set_postfix(status=record["status"])
    progress.close()

    metrics = client.metrics()
    successes = sum(record["status"] == "success" for record in records)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "framework": "agentic_opencv_f2g_oracle_actions",
        "tasks": tasks,
        "episode_ids": episode_ids,
        "num_repeats": args.num_repeats,
        "episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records) if records else 0,
        "status_counts": {
            status: sum(record["status"] == status for record in records)
            for status in sorted({record["status"] for record in records})
        },
        "environments": {
            "vlm": "robomme-qwen38",
            "f2g": "f2g",
            "simulation": "robomme",
        },
        "gpu_before_eval": pre_eval_gpu,
        "vlm": metrics,
        "f2g": f2g_health,
        "depth_scale": args.depth_scale,
    }
    _write_json(run_dir / "summary.json", summary)
    client.close()


if __name__ == "__main__":
    main()
