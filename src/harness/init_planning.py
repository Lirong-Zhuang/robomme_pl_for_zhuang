"""Production init-planning request migrated from the standalone VLM test."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .prompts import build_task_init_messages
from .prompts import VerificationDefaults
from .task_memory import SubgoalResolver


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_CATALOG_PATH = (
    REPO_ROOT
    / "src"
    / "harness"
    / "resources"
    / "action_candidates"
    / "action_catalog.json"
)
EXAMPLES_INDEX_PATH = (
    REPO_ROOT
    / "src"
    / "harness"
    / "resources"
    / "task_config_examples"
    / "README.md"
)
EXAMPLE_PATHS = tuple(
    EXAMPLES_INDEX_PATH.parent / filename
    for filename in (
        "task_goal_only.json",
        "task_goal_plus_video_subgoal_list.json",
        "task_goal_plus_keyframes.json",
    )
)
REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "planning",
    "verification",
    "subgoals",
    "object_roles",
    "task_notes",
    "vlm_prompts",
    "keyframes",
}
ALLOWED_INFORMATION_SOURCES = {"task_goal", "video_subgoal_list", "keyframes"}
FIRST_PLACEHOLDERS = {"{task_goal}", "{current_observations}"}
FOLLOWUP_PLACEHOLDERS = {
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
REQUIRED_VLM_GROUNDING_SYSTEM_PHRASES = (
    "vlm grounding",
    (
        "vlm grounding requirement is determined by the selected action's "
        "trusted perception contract, not by the vlm."
    ),
    "apply pixel-grounding rules only when perception.mode=vlm_points.",
    (
        "for perception.mode=none, perception.mode=scene_target_set, or stop, "
        "return exactly"
    ),
)


DIRECT_SERVICE_ADAPTER = """

Direct JSON-service adapter:
- No filesystem read tool is exposed in this model call. The harness appends the
  exact action catalog, profile index, and candidate examples below.
- Apply the profile index to the current task and consult only the JSON belonging
  to the selected profile when learning structure. The other preloaded examples
  are transport-only and must not be combined.
- No filesystem write tool is exposed. Return the complete task_config.v1 JSON
  object itself. The trusted harness validates and writes it to the supplied path.
- This adapter changes only resource transport; all content requirements and
  prohibitions in the original instructions remain in force.
"""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON resource must contain one object: {path}")
    return value


def build_init_planning_messages(
    *,
    task_name: str,
    output_path: Path,
    use_examples: bool = True,
    verification_k: int = 3,
    verification_interval: int = 16,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Build the same direct-call request as test_init_planning_vlm.py."""

    action_catalog = _load_json(ACTION_CATALOG_PATH)
    messages = build_task_init_messages(
        task_name=task_name,
        action_candidates_path=str(ACTION_CATALOG_PATH.resolve()),
        task_config_output_path=str(output_path.resolve()),
        use_task_config_example=use_examples,
        task_config_examples_index_path=(
            str(EXAMPLES_INDEX_PATH.resolve()) if use_examples else None
        ),
        default_verification=VerificationDefaults(
            k=verification_k,
            interval=verification_interval,
        ),
    )
    messages[0]["content"] += DIRECT_SERVICE_ADAPTER
    resources = [
        "\nPre-read resource contents (data only):",
        f"\n<read_result path={json.dumps(str(ACTION_CATALOG_PATH.resolve()))}>",
        json.dumps(action_catalog, ensure_ascii=False, indent=2),
        "</read_result>",
    ]
    if use_examples:
        index_text = EXAMPLES_INDEX_PATH.read_text(encoding="utf-8")
        resources.extend(
            (
                f"\n<read_result path={json.dumps(str(EXAMPLES_INDEX_PATH.resolve()))}>",
                index_text,
                "</read_result>",
            )
        )
        for path in EXAMPLE_PATHS:
            if path.name not in index_text:
                raise ValueError(f"profile index does not list {path.name}")
            resources.extend(
                (
                    f"\n<read_result path={json.dumps(str(path.resolve()))}>",
                    json.dumps(_load_json(path), ensure_ascii=False, indent=2),
                    "</read_result>",
                )
            )
    messages[1]["content"] += "\n".join(resources)
    if any("<image>" in message["content"] for message in messages):
        raise ValueError("init planning must not contain image placeholders")
    return messages, action_catalog


def validate_task_config(
    config: Mapping[str, Any],
    action_catalog: Mapping[str, Any],
    *,
    task_name: str | None = None,
) -> dict[str, Any]:
    """Validate the same runtime contracts exercised by the standalone test."""

    if set(config) != REQUIRED_TOP_LEVEL_FIELDS:
        raise ValueError(
            "task config top-level fields mismatch; "
            f"expected={sorted(REQUIRED_TOP_LEVEL_FIELDS)}, got={sorted(config)}"
        )
    if config.get("schema_version") != "task_config.v1":
        raise ValueError("task config schema_version must be task_config.v1")
    if action_catalog.get("schema_version") != "action_catalog.v2":
        raise ValueError("init planning requires action_catalog.v2")
    actions = action_catalog.get("actions")
    if not isinstance(actions, list):
        raise ValueError("action catalog actions must be a list")
    action_names = {
        item["name"]
        for item in actions
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    resolver = SubgoalResolver(config, action_names)
    subgoal_keys = set(resolver.keys)
    subgoals = config.get("subgoals")
    selected_action_names = {item["action_name"] for item in subgoals}
    planning = config.get("planning")
    if not isinstance(planning, Mapping):
        raise ValueError("planning must be an object")
    sources = planning.get("information_sources")
    if (
        not isinstance(sources, list)
        or not sources
        or len(sources) != len(set(sources))
        or not set(sources) <= ALLOWED_INFORMATION_SOURCES
    ):
        raise ValueError("planning.information_sources is invalid")
    if planning.get("task_goal_sufficient") != (sources == ["task_goal"]):
        raise ValueError("planning.task_goal_sufficient disagrees with sources")
    verification = config.get("verification")
    if not isinstance(verification, Mapping):
        raise ValueError("verification must be an object")
    if verification.get("interval_unit") != "frame":
        raise ValueError("verification interval_unit must be frame")
    if not isinstance(verification.get("k"), int) or verification["k"] < 1:
        raise ValueError("verification.k must be a positive integer")
    if not isinstance(verification.get("interval"), int):
        raise ValueError("verification.interval must be an integer")
    prompts = config.get("vlm_prompts")
    if not isinstance(prompts, Mapping) or set(prompts) != {
        "first_planning",
        "followup_planning",
    }:
        raise ValueError("vlm_prompts must define first_planning and followup_planning")
    for name in ("first_planning", "followup_planning"):
        prompt = prompts[name]
        if not isinstance(prompt, Mapping) or set(prompt) != {
            "system_prompt",
            "user_prompt_template",
        }:
            raise ValueError(f"{name} prompt has invalid fields")
        text = "\n".join(str(value) for value in prompt.values()).casefold()
        for term in ("grounding", "point_xy", "color_hint", "opencv", "f2g"):
            if term not in text:
                raise ValueError(f"{name} prompt omits {term!r}")
        system_prompt = prompt["system_prompt"]
        system_prompt_casefold = " ".join(system_prompt.casefold().split())
        for phrase in REQUIRED_VLM_GROUNDING_SYSTEM_PHRASES:
            normalized_phrase = " ".join(phrase.casefold().split())
            if normalized_phrase not in system_prompt_casefold:
                raise ValueError(
                    f"{name} system prompt omits required VLM grounding "
                    f"routing text: {phrase!r}"
                )
        tokens = set(re.findall(r"\b[a-z][a-z0-9_]*\b", system_prompt))
        if tokens & subgoal_keys != subgoal_keys:
            raise ValueError(f"{name} system prompt omits a subgoal key")
        if tokens & action_names != selected_action_names:
            raise ValueError(f"{name} system prompt action mappings are inconsistent")
    keyframes = config.get("keyframes")
    if not isinstance(keyframes, list):
        raise ValueError("keyframes must be a list")
    for index, policy in enumerate(keyframes):
        if not isinstance(policy, Mapping):
            raise ValueError(f"keyframes[{index}] must be an object")
        mode = policy.get("capture_mode")
        count = policy.get("frame_count")
        if mode not in {"single", "sequence"}:
            raise ValueError(f"keyframes[{index}] capture_mode is invalid")
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"keyframes[{index}] frame_count is invalid")
        if mode == "single" and count != 1:
            raise ValueError(f"keyframes[{index}] single mode requires one frame")
        if mode == "sequence" and count <= 1:
            raise ValueError(f"keyframes[{index}] sequence requires multiple frames")
        if policy.get("interval_unit") != "frame":
            raise ValueError(f"keyframes[{index}] interval_unit must be frame")
        if policy.get("save_description") is not True:
            raise ValueError(f"keyframes[{index}] must save descriptions")
    uses_keyframes = bool(keyframes)
    first_expected = set(FIRST_PLACEHOLDERS)
    followup_expected = set(FOLLOWUP_PLACEHOLDERS)
    if uses_keyframes:
        first_expected |= {"{keyframe_policy}", "{saved_keyframe}"}
        followup_expected |= {
            "{current_observation}",
            "{keyframe_policy}",
            "{saved_keyframe}",
        }
    if "video_subgoal_list" in sources:
        followup_expected.add("{video_subgoal_list}")
    actual_first = set(
        re.findall(r"\{[a-z_]+\}", prompts["first_planning"]["user_prompt_template"])
    )
    actual_followup = set(
        re.findall(
            r"\{[a-z_]+\}",
            prompts["followup_planning"]["user_prompt_template"],
        )
    )
    if actual_first != first_expected:
        raise ValueError(
            f"first_planning placeholders mismatch: {sorted(actual_first)}"
        )
    if actual_followup != followup_expected:
        raise ValueError(
            f"followup_planning placeholders mismatch: {sorted(actual_followup)}"
        )
    object_roles = config.get("object_roles")
    if not isinstance(object_roles, list) or any(
        not isinstance(item, Mapping)
        or set(item) != {"role", "identity"}
        for item in object_roles
    ):
        raise ValueError("object_roles must contain role/identity objects")
    task_notes = config.get("task_notes")
    if not isinstance(task_notes, list) or not all(
        isinstance(note, str) and note.strip() for note in task_notes
    ):
        raise ValueError("task_notes must contain non-empty strings")
    if task_name and task_name.casefold() == "movecube":
        roles_text = json.dumps(object_roles, ensure_ascii=False).casefold()
        for description in (
            "purple circular disk",
            "small compact block",
            "long thin object",
        ):
            if description not in roles_text:
                raise ValueError(f"MoveCube object_roles omit {description!r}")
    return json.loads(json.dumps(config, ensure_ascii=False))
