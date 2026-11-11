"""Standalone init-planning VLM test.

Required environment (the same VLM environment used by Oracle Plus):
    micromamba activate robomme-qwen38

Run from the repository root after activating that environment:
    python src/harness/tests/test_init_planning_vlm.py

The environment must provide CUDA, PyTorch, Transformers, and a recent ms-swift
with Qwen3.8 support. Set QWEN_MODEL_PATH if the default checkpoint path below is
not valid on the current machine. This test needs no simulator or VLA server.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from harness.prompts import build_task_init_messages  # noqa: E402
from oracle_plus.vlm import VLMSubgoalPlanner27B  # noqa: E402


# -----------------------------------------------------------------------------
# Test settings: edit this block before running.
# -----------------------------------------------------------------------------

TASK_NAME = "BinFill"  # BinFill, MoveCube, or PickHighlight
USE_TASK_CONFIG_EXAMPLE = True

OUTPUT_DIRECTORY = REPO_ROOT / "runs" / "agentic_method" / "tests" / "init_planning"
OUTPUT_FILE_NAME = "binfill_task_config_with_example_v1.json"
ACTION_CANDIDATES_PATH = (
    REPO_ROOT
    / "src"
    / "harness"
    / "resources"
    / "action_candidates"
    / "action_catalog.json"
)
# Keep this independently editable. For a fair inference test, prefer an example
# from a different task rather than the gold config for TASK_NAME.
TASK_CONFIG_EXAMPLE_PATH = (
    REPO_ROOT
    / "src"
    / "harness"
    / "resources"
    / "task_config_candidates"
    / "binfill.json"
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
- The Python harness has already executed the required read operations. The exact
  read results are appended to the user message below and must be treated as the
  contents returned by the read tool for their declared paths.
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
    "task_notes",
    "vlm_prompts",
    "keyframes",
}
_ALLOWED_INFORMATION_SOURCES = {
    "task_goal",
    "video_subgoal_list",
    "keyframes",
}
_VERIFICATION_PLACEHOLDERS = {
    "{task_goal}",
    "{current_subgoal}",
    "{action_name}",
    "{acceptance_criteria}",
    "{verification_parameters}",
    "{task_notes}",
    "{subgoal_start_observations}",
    "{recent_observations}",
    "{task_log_summary}",
}
_PLANNING_PLACEHOLDERS = {
    "{task_goal}",
    "{planning_policy}",
    "{subgoal_catalog}",
    "{task_notes}",
    "{keyframe_policy}",
    "{task_log}",
    "{robot_state}",
    "{current_observations}",
    "{current_subgoal}",
    "{current_subgoal_sequence}",
    "{verification_result}",
}


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON resource does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON resource must contain one object: {path}")
    return value


def _build_direct_messages(
    *,
    output_path: Path,
    action_catalog: dict[str, Any],
    config_example: dict[str, Any] | None,
) -> list[dict[str, str]]:
    messages = build_task_init_messages(
        task_name=TASK_NAME,
        action_candidates_path=str(ACTION_CANDIDATES_PATH.resolve()),
        task_config_output_path=str(output_path),
        use_task_config_example=USE_TASK_CONFIG_EXAMPLE,
        task_config_example_path=(
            str(TASK_CONFIG_EXAMPLE_PATH.resolve())
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
    if config_example is not None:
        resource_results.extend(
            (
                f"\n<read_result path={json.dumps(str(TASK_CONFIG_EXAMPLE_PATH.resolve()))}>",
                json.dumps(config_example, ensure_ascii=False, indent=2),
                "</read_result>",
            )
        )
    messages[1]["content"] += "\n".join(resource_results)
    return messages


def _placeholders(template: str) -> set[str]:
    return set(re.findall(r"\{[a-z_]+\}", template))


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
    for index, subgoal in enumerate(subgoals):
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

    prompts = config["vlm_prompts"]
    verifier_template = prompts["verification"]["user_prompt_template"]
    planner_template = prompts["in_task_planning"]["user_prompt_template"]
    if _placeholders(verifier_template) != _VERIFICATION_PLACEHOLDERS:
        raise ValueError("verification prompt has incorrect runtime placeholders")
    if _placeholders(planner_template) != _PLANNING_PLACEHOLDERS:
        raise ValueError("in-task-planning prompt has incorrect runtime placeholders")

    for index, keyframe in enumerate(config["keyframes"]):
        if keyframe.get("interval_unit") != "frame":
            raise ValueError(f"keyframes[{index}].interval_unit must be 'frame'")


def main() -> None:
    output_path = (OUTPUT_DIRECTORY / OUTPUT_FILE_NAME).resolve()
    action_catalog = _load_json_object(ACTION_CANDIDATES_PATH)
    config_example = (
        _load_json_object(TASK_CONFIG_EXAMPLE_PATH)
        if USE_TASK_CONFIG_EXAMPLE
        else None
    )
    messages = _build_direct_messages(
        output_path=output_path,
        action_catalog=action_catalog,
        config_example=config_example,
    )

    print(f"Task: {TASK_NAME}")
    print(f"Prompt variant: {'with' if USE_TASK_CONFIG_EXAMPLE else 'without'} config example")
    print(f"Model: {MODEL_PATH.expanduser().resolve()}")
    print(f"Output: {output_path}")
    print("Loading VLM...")
    model = VLMSubgoalPlanner27B(
        model_path=str(MODEL_PATH),
        attn_impl=ATTN_IMPL,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
    config, raw_response, vlm_latency_ms, request_to_json_ms = model.infer_json_object(
        messages=messages
    )
    try:
        _validate_task_config(config, action_catalog)
    except (KeyError, TypeError, ValueError):
        print("VLM returned JSON, but task-config validation failed. Raw response:")
        print(raw_response)
        raise

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"VLM latency: {vlm_latency_ms:.1f} ms")
    print(f"Request-to-validated-JSON latency: {request_to_json_ms:.1f} ms")
    print(f"Task config written: {output_path}")


if __name__ == "__main__":
    main()
