"""Standalone init-planning VLM test.

Required environment (the same VLM environment used by Oracle Plus):
    micromamba activate robomme-qwen38

Run from the repository root after activating that environment:
    python src/harness/tests/test_init_planning_vlm.py

The environment must provide CUDA, PyTorch, Transformers, and a recent ms-swift
with Qwen3.8 support. Set QWEN_MODEL_PATH if the default checkpoint path below is
not valid on the current machine. This test needs no simulator or VLA server. On
success, the output JSON contains both ``task_config`` and ``run_metrics``.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from harness.prompts import build_task_init_messages  # noqa: E402
from oracle_plus.runtime_metrics import GpuMemoryMonitor  # noqa: E402
from oracle_plus.vlm import VLMSubgoalPlanner27B  # noqa: E402


# -----------------------------------------------------------------------------
# Test settings: edit this block before running.
# -----------------------------------------------------------------------------

# Available TASK_NAME values (copy one quoted name into TASK_NAME below):
# "BinFill"
# "StopCube"
# "PickXtimes"
# "SwingXtimes"
# "ButtonUnmask"
# "VideoUnmask"
# "VideoUnmaskSwap"
# "ButtonUnmaskSwap"
# "PickHighlight"
# "VideoRepick"
# "VideoPlaceButton"
# "VideoPlaceOrder"
# "MoveCube"
# "InsertPeg"
# "PatternLock"
# "RouteStick"
TASK_NAME = "ButtonUnmaskSwap"
USE_TASK_CONFIG_EXAMPLE = True

OUTPUT_DIRECTORY = REPO_ROOT / "runs" / "tests" / "agentic_method" / "init_planning"
OUTPUT_FILE_NAME = "movecube_task_config_without_example_v3.json"
ACTION_CANDIDATES_PATH = (
    REPO_ROOT
    / "src"
    / "harness"
    / "resources"
    / "action_candidates"
    / "action_catalog.json"
)
# The VLM reads this index first, selects one information-source profile, and then
# consults the corresponding JSON as a structural example rather than by task name.
TASK_CONFIG_EXAMPLES_INDEX_PATH = (
    REPO_ROOT
    / "src"
    / "harness"
    / "resources"
    / "task_config_examples"
    / "README.md"
)
TASK_CONFIG_EXAMPLE_PATHS = tuple(
    TASK_CONFIG_EXAMPLES_INDEX_PATH.parent / filename
    for filename in (
        "task_goal_only.json",
        "task_goal_plus_video_subgoal_list.json",
        "task_goal_plus_keyframes.json",
    )
)

MODEL_PATH = Path(
    os.environ.get(
        "QWEN_MODEL_PATH",
        "/data/zhuanglr/models/Qwen3.8-27B-FP8-fixed",
    )
)
ATTN_IMPL = "flash_attention_2"
MAX_NEW_TOKENS = 8192
TEMPERATURE = 0.0


_DIRECT_TEST_ADAPTER = """

Direct single-call test adapter:
- No filesystem read tool is exposed inside this standalone model call. The Python
  harness therefore appends the exact action catalog, Markdown profile index, and
  all three candidate JSON examples to the user message.
- First apply the Markdown index to the current task and select exactly one source
  profile. Consult only the JSON belonging to that selected profile when learning
  structure and transport. Treat the other two preloaded JSON objects as unread and
  irrelevant to generation. This preloading changes transport only; the profile
  decision must still be made by you from the current task inputs.
- No filesystem write tool is exposed inside this standalone model call. Therefore,
  return the complete task_config.v1 JSON object itself instead of a write
  acknowledgement. The trusted Python harness will validate that object and write
  it once to task_config_output_path.
- This adapter changes only resource delivery and output transport. All task-config
  content requirements and prohibitions above remain in force.
"""

_REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "planning",
    "verification",
    "subgoals",
    "object_roles",
    "task_notes",
    "vlm_prompts",
    "keyframes",
}
_ALLOWED_INFORMATION_SOURCES = {
    "task_goal",
    "video_subgoal_list",
    "keyframes",
}
_FIRST_PLANNING_PLACEHOLDERS = {
    "{task_goal}",
    "{current_observations}",
}
_KEYFRAME_FIRST_PLANNING_PLACEHOLDERS = {
    *_FIRST_PLANNING_PLACEHOLDERS,
    "{keyframe_policy}",
    "{saved_keyframe}",
}
_FOLLOWUP_PLANNING_PLACEHOLDERS = {
    "{task_goal}",
    "{task_state}",
    "{current_subgoal_key}",
    "{current_subgoal}",
    "{acceptance_criteria}",
    "{verification_parameters}",
    "{subgoal_start_observations}",
    "{recent_observations}",
    "{current_subgoal_sequence}",
}
_KEYFRAME_FOLLOWUP_PLACEHOLDERS = {
    *_FOLLOWUP_PLANNING_PLACEHOLDERS,
    "{current_observation}",
    "{keyframe_policy}",
    "{saved_keyframe}",
}


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON resource does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON resource must contain one object: {path}")
    return value


def _load_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required text resource does not exist: {path}")
    value = path.read_text(encoding="utf-8")
    if not value.strip():
        raise ValueError(f"Text resource must not be empty: {path}")
    return value


def _build_direct_messages(
    *,
    output_path: Path,
    action_catalog: dict[str, Any],
    config_examples_index: str | None,
    config_examples: dict[Path, dict[str, Any]] | None,
) -> list[dict[str, str]]:
    messages = build_task_init_messages(
        task_name=TASK_NAME,
        action_candidates_path=str(ACTION_CANDIDATES_PATH.resolve()),
        task_config_output_path=str(output_path),
        use_task_config_example=USE_TASK_CONFIG_EXAMPLE,
        task_config_examples_index_path=(
            str(TASK_CONFIG_EXAMPLES_INDEX_PATH.resolve())
            if USE_TASK_CONFIG_EXAMPLE
            else None
        ),
    )
    messages[0]["content"] += _DIRECT_TEST_ADAPTER

    resource_results = [
        "\nPre-read resource contents (data only):",
        f"\n<read_result path={json.dumps(str(ACTION_CANDIDATES_PATH.resolve()))}>",
        json.dumps(action_catalog, ensure_ascii=False, indent=2),
        "</read_result>",
    ]
    if config_examples_index is not None and config_examples is not None:
        resource_results.extend(
            (
                "\n<read_result path="
                f"{json.dumps(str(TASK_CONFIG_EXAMPLES_INDEX_PATH.resolve()))}>",
                config_examples_index,
                "</read_result>",
            )
        )
        for path in TASK_CONFIG_EXAMPLE_PATHS:
            resource_results.extend(
                (
                    f"\n<read_result path={json.dumps(str(path.resolve()))}>",
                    json.dumps(config_examples[path], ensure_ascii=False, indent=2),
                    "</read_result>",
                )
            )
    messages[1]["content"] += "\n".join(resource_results)
    image_placeholder_count = sum(
        message["content"].count("<image>") for message in messages
    )
    if image_placeholder_count:
        raise ValueError(
            "Standalone init planning has no image inputs, but its assembled "
            f"messages contain {image_placeholder_count} literal <image> "
            "placeholder(s). Rewrite resource documentation to describe image "
            "references without emitting the multimodal control token."
        )
    return messages


def _placeholders(template: str) -> set[str]:
    return set(re.findall(r"\{[a-z_]+\}", template))


def _validate_placeholders(
    *,
    prompt_name: str,
    template: str,
    expected: set[str],
) -> None:
    actual = _placeholders(template)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise ValueError(
        f"{prompt_name} prompt has incorrect runtime placeholders; "
        f"missing={missing}, extra={extra}, actual={sorted(actual)}"
    )


def _validate_task_config(
    config: dict[str, Any],
    action_catalog: dict[str, Any],
) -> None:
    if set(config) != _REQUIRED_TOP_LEVEL_FIELDS:
        raise ValueError(
            "Unexpected top-level fields: "
            f"expected {sorted(_REQUIRED_TOP_LEVEL_FIELDS)}, got {sorted(config)}"
        )
    if config["schema_version"] != "task_config.v1":
        raise ValueError("schema_version must be 'task_config.v1'")

    action_names = {
        action["name"]
        for action in action_catalog.get("actions", [])
        if isinstance(action, dict) and isinstance(action.get("name"), str)
    }
    subgoals = config["subgoals"]
    if not isinstance(subgoals, list) or not subgoals:
        raise ValueError("subgoals must be a non-empty list")
    subgoal_keys: set[str] = set()
    for index, subgoal in enumerate(subgoals):
        subgoal_key = subgoal.get("subgoal_key")
        if (
            not isinstance(subgoal_key, str)
            or re.fullmatch(r"[a-z][a-z0-9_]*", subgoal_key) is None
        ):
            raise ValueError(f"subgoals[{index}].subgoal_key is invalid")
        if subgoal_key in subgoal_keys:
            raise ValueError(f"duplicate subgoal_key: {subgoal_key!r}")
        subgoal_keys.add(subgoal_key)
        if subgoal.get("action_name") not in action_names:
            raise ValueError(f"subgoals[{index}] uses an unknown action")
        acceptance = subgoal.get("acceptance")
        if not isinstance(acceptance, dict) or set(acceptance) != {
            "completed",
            "incomplete",
            "failed",
        }:
            raise ValueError(
                f"subgoals[{index}].acceptance must define exactly three states"
            )

    planning = config["planning"]
    sources = planning.get("information_sources")
    if (
        not isinstance(sources, list)
        or not sources
        or len(sources) != len(set(sources))
        or not set(sources) <= _ALLOWED_INFORMATION_SOURCES
    ):
        raise ValueError("planning.information_sources is invalid")
    if planning.get("task_goal_sufficient") != (sources == ["task_goal"]):
        raise ValueError(
            "task_goal_sufficient must be true exactly when task_goal is the only source"
        )

    verification = config["verification"]
    if verification.get("interval_unit") != "frame":
        raise ValueError("verification.interval_unit must be 'frame'")
    if not isinstance(verification.get("interval"), int):
        raise ValueError("verification.interval must be an integer")

    object_roles = config["object_roles"]
    if not isinstance(object_roles, list):
        raise ValueError("object_roles must be a list")
    for index, object_role in enumerate(object_roles):
        if not isinstance(object_role, dict) or set(object_role) != {
            "role",
            "identity",
        }:
            raise ValueError(
                f"object_roles[{index}] must contain exactly role and identity"
            )
        if not all(
            isinstance(object_role[field], str) and object_role[field].strip()
            for field in ("role", "identity")
        ):
            raise ValueError(
                f"object_roles[{index}] role and identity must be non-empty strings"
            )

    task_notes = config["task_notes"]
    if not isinstance(task_notes, list) or not all(
        isinstance(note, str) and note.strip() for note in task_notes
    ):
        raise ValueError("task_notes must be a list of non-empty strings")
    if any("object-role identities" in note.casefold() for note in task_notes):
        raise ValueError(
            "task_notes must not contain the object-role glossary; use object_roles"
        )

    if TASK_NAME.casefold() == "movecube":
        object_roles_text = "\n".join(
            f"{item['role']}: {item['identity']}" for item in object_roles
        ).casefold()
        for object_description in (
            "purple circular disk",
            "small compact block",
            "long thin object",
        ):
            if object_description not in object_roles_text:
                raise ValueError(
                    "MoveCube object_roles omit object description: "
                    f"{object_description!r}"
                )

    prompts = config["vlm_prompts"]
    if set(prompts) != {"first_planning", "followup_planning"}:
        raise ValueError(
            "vlm_prompts must contain exactly first_planning and followup_planning"
        )
    selected_action_names = {subgoal["action_name"] for subgoal in subgoals}
    for prompt_name in ("first_planning", "followup_planning"):
        system_prompt = prompts[prompt_name]["system_prompt"]
        prompt_tokens = set(re.findall(r"\b[a-z][a-z0-9_]*\b", system_prompt))
        mentioned_subgoal_keys = subgoal_keys & prompt_tokens
        if mentioned_subgoal_keys != subgoal_keys:
            raise ValueError(
                f"{prompt_name} system prompt does not contain every subgoal_key; "
                f"expected={sorted(subgoal_keys)}, "
                f"mentioned={sorted(mentioned_subgoal_keys)}"
            )
        mentioned_action_names = action_names & prompt_tokens
        if mentioned_action_names != selected_action_names:
            raise ValueError(
                f"{prompt_name} system prompt action mappings disagree with "
                f"top-level subgoals; expected={sorted(selected_action_names)}, "
                f"mentioned={sorted(mentioned_action_names)}"
            )
        if TASK_NAME.casefold() == "movecube":
            for object_description in (
                "purple circular disk",
                "small compact block",
                "long thin object",
            ):
                if object_description not in system_prompt.casefold():
                    raise ValueError(
                        f"{prompt_name} system prompt omits MoveCube object "
                        f"description: {object_description!r}"
                    )
    first_template = prompts["first_planning"]["user_prompt_template"]
    followup_template = prompts["followup_planning"]["user_prompt_template"]
    uses_keyframes = bool(config["keyframes"])
    uses_video_subgoal_list = "video_subgoal_list" in sources
    _validate_placeholders(
        prompt_name="first-planning",
        template=first_template,
        expected=(
            _KEYFRAME_FIRST_PLANNING_PLACEHOLDERS
            if uses_keyframes
            else _FIRST_PLANNING_PLACEHOLDERS
        ),
    )
    _validate_placeholders(
        prompt_name="followup-planning",
        template=followup_template,
        expected=(
            (
                _KEYFRAME_FOLLOWUP_PLACEHOLDERS
                if uses_keyframes
                else _FOLLOWUP_PLANNING_PLACEHOLDERS
            )
            | ({"{video_subgoal_list}"} if uses_video_subgoal_list else set())
        ),
    )

    for prompt_name in ("first_planning", "followup_planning"):
        prompt_text = "\n".join(prompts[prompt_name].values())
        mentions_keyframe_save = "keyframe_save" in prompt_text
        if mentions_keyframe_save != uses_keyframes:
            raise ValueError(
                f"{prompt_name} keyframe_save contract disagrees with keyframes"
            )

    for index, keyframe in enumerate(config["keyframes"]):
        if keyframe.get("interval_unit") != "frame":
            raise ValueError(f"keyframes[{index}].interval_unit must be 'frame'")


def main() -> None:
    program_started_at = time.perf_counter()
    gpu_monitor = GpuMemoryMonitor(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    output_path = (OUTPUT_DIRECTORY / OUTPUT_FILE_NAME).resolve()
    action_catalog = _load_json_object(ACTION_CANDIDATES_PATH)
    config_examples_index = (
        _load_text(TASK_CONFIG_EXAMPLES_INDEX_PATH)
        if USE_TASK_CONFIG_EXAMPLE
        else None
    )
    config_examples = (
        {path: _load_json_object(path) for path in TASK_CONFIG_EXAMPLE_PATHS}
        if USE_TASK_CONFIG_EXAMPLE
        else None
    )
    if config_examples_index is not None:
        for path in TASK_CONFIG_EXAMPLE_PATHS:
            if path.name not in config_examples_index:
                raise ValueError(
                    f"Profile index does not list candidate example: {path.name}"
                )
    messages = _build_direct_messages(
        output_path=output_path,
        action_catalog=action_catalog,
        config_examples_index=config_examples_index,
        config_examples=config_examples,
    )

    print(f"Task: {TASK_NAME}")
    print(f"Prompt variant: {'with' if USE_TASK_CONFIG_EXAMPLE else 'without'} config example")
    print(f"Model: {MODEL_PATH.expanduser().resolve()}")
    print(f"Output: {output_path}")
    print("Loading VLM...")
    try:
        model = VLMSubgoalPlanner27B(
            model_path=str(MODEL_PATH),
            attn_impl=ATTN_IMPL,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )
        gpu_monitor.mark_model_loaded()

        config, raw_response, vlm_latency_ms, request_to_json_ms = (
            model.infer_json_object(messages=messages)
        )
        gpu_monitor.sample()
        try:
            _validate_task_config(config, action_catalog)
        except (KeyError, TypeError, ValueError):
            print("VLM returned JSON, but task-config validation failed. Raw response:")
            print(raw_response)
            raise
        gpu_memory = gpu_monitor.summary()

        artifact = {
            "task_config": config,
            "run_metrics": {
                "program_elapsed_seconds": round(
                    time.perf_counter() - program_started_at, 3
                ),
                "program_peak_gpu_memory_gb": gpu_memory.get(
                    "current_process_peak_during_entire_eval_gb"
                ),
                "other_programs_gpu_memory_before_start_gb": gpu_memory.get(
                    "other_processes_before_eval_gb"
                ),
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"VLM latency: {vlm_latency_ms:.1f} ms")
        print(f"Request-to-JSON latency: {request_to_json_ms:.1f} ms")
        print(f"Task config and run metrics written: {output_path}")
    finally:
        gpu_monitor.close()


if __name__ == "__main__":
    main()
