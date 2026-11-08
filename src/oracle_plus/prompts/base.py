"""Shared subgoal-and-point prompts for task-specific 27B inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal


CallType = Literal["initial", "followup"]
InitialObservationMode = Literal["current", "demonstration"]


SYSTEM_TEMPLATE = """You are a helpful assistant that guides a robot by predicting
the next ordinary-language subgoal and, only when needed, its target point.
The controller maps your subgoal to a trusted robot function; never output a
function name or an action sequence.

Possible subgoals:
{subgoals}

Task-specific notes:
{notes}

Example:
{example}

Consistency rules:
1. Use one of the supplied subgoal forms. Do not invent a robot function.
2. If the current subgoal is incomplete, keep the same next_subgoal.
3. If it is complete, advance next_subgoal to the next stage.
4. Only when the entire task is complete, output next_subgoal as STOP.
{point_rules}
7. {description_rule}

Output format:
For the initial call, return only:
{initial_schema}

For a follow-up call, return only:
{followup_schema}

Return JSON only. Start the answer directly with {{ and do not restate the task.
Do not output analysis, reasoning, think tags, action_name, or action_sequence.
"""


INITIAL_SCHEMA_WITH_DESCRIPTION = """```json
{
  "description": "visible initial-state evidence only; at most 50 English words and two short sentences",
  "subgoal_sequence": ["ordered subgoal 1", "ordered subgoal 2"],
  "completed": false,
  "next_subgoal": "the first subgoal",
  "point": [y, x] or null
}
```"""

FOLLOWUP_SCHEMA_WITH_DESCRIPTION = """```json
{
  "description": "visible completion evidence only; at most 50 English words and two short sentences",
  "completed": true or false,
  "next_subgoal": "the next subgoal, or the current one if it is incomplete",
  "point": [y, x] or null
}
```"""

INITIAL_SCHEMA_WITHOUT_DESCRIPTION = """```json
{
  "subgoal_sequence": ["ordered subgoal 1", "ordered subgoal 2"],
  "completed": false,
  "next_subgoal": "the first subgoal",
  "point": [y, x] or null
}
```"""

FOLLOWUP_SCHEMA_WITHOUT_DESCRIPTION = """```json
{
  "completed": true or false,
  "next_subgoal": "the next subgoal, or the current one if it is incomplete",
  "point": [y, x] or null
}
```"""


INITIAL_USER_TEMPLATE = """The task goal is: {task_goal}
This is the initial call. Observe the initial state, predict the complete subgoal plan,
then return its first next_subgoal and target point if that subgoal is a pickup.
The overall image resolution is height={image_height}, width={image_width} pixels.
Initial overall image: <image>
Initial wrist image: <image>
Any target point must refer to the initial overall image in original pixel
coordinates. Do not use normalized 0-1000 coordinates.
Start directly with the JSON object. Output JSON only.
"""


INITIAL_USER_TEMPLATE_OVERALL_ONLY = """The task goal is: {task_goal}
This is the initial call. Observe the initial state, predict the complete subgoal plan,
then return its first next_subgoal and target point if that subgoal is a pickup.
The overall image resolution is height={image_height}, width={image_width} pixels.
Initial overall image: <image>
Any target point must refer to the initial overall image in original pixel
coordinates. Do not use normalized 0-1000 coordinates.
Start directly with the JSON object. Output JSON only.
"""


FOLLOWUP_USER_TEMPLATE = """The task goal is: {task_goal}
The history of previously active subgoals is: {history}
The current subgoal is: {current_subgoal}
The overall image resolution is height={image_height}, width={image_width} pixels.

Subgoal-init paired observation, captured immediately before this current subgoal
first became active and before any attempt to execute it:
  Subgoal-init overall: <image>
  Subgoal-init wrist: <image>

Recent sliding-window paired observations, oldest to newest. These are continuous
recent observations captured during or after execution of the current subgoal:
{observation_lines}

Compare the fixed subgoal-init observation with the recent sliding-window
observations to decide whether the current subgoal is complete; do not infer success
merely because the robot stopped moving. Then return the subgoal that should be
active now. If incomplete, keep the same subgoal. Any target point must refer to the
newest sliding-window overall image in original pixel coordinates. Do not use
normalized 0-1000 coordinates.
Start directly with the JSON object. Output JSON only.
"""


FOLLOWUP_USER_TEMPLATE_OVERALL_ONLY = """The task goal is: {task_goal}
The history of previously active subgoals is: {history}
The current subgoal is: {current_subgoal}
The overall image resolution is height={image_height}, width={image_width} pixels.

Subgoal-init overall observation, captured immediately before this current subgoal
first became active and before any attempt to execute it:
  Subgoal-init overall: <image>

Recent sliding-window overall observations, oldest to newest. These are continuous
recent observations captured during or after execution of the current subgoal:
{observation_lines}

Compare the fixed subgoal-init observation with the recent sliding-window
observations to decide whether the current subgoal is complete; do not infer success
merely because the robot stopped moving. Then return the subgoal that should be
active now. If incomplete, keep the same subgoal. Any target point must refer to the
newest sliding-window overall image in original pixel coordinates. Do not use
normalized 0-1000 coordinates.
Start directly with the JSON object. Output JSON only.
"""


INITIAL_DEMONSTRATION_USER_TEMPLATE = """The task goal is: {task_goal}
This is the initial call. The observations below contain sampled frames from a
pre-recorded demonstration, oldest to newest, followed by the initial state of
the new execution. Infer the demonstrated strategy, predict the complete
subgoal plan for the new execution, then return its first next_subgoal.
The overall image resolution is height={image_height}, width={image_width} pixels.

{observation_lines}

Do not treat actions in the demonstration as progress in the new execution.
Any target point must use original pixel coordinates from the top-left, not
normalized 0-1000 coordinates.
Start directly with the JSON object. Output JSON only.
"""


INITIAL_DEMONSTRATION_USER_TEMPLATE_OVERALL_ONLY = """The task goal is: {task_goal}
This is the initial call. The observations below contain sampled overall-view
frames from a pre-recorded demonstration, oldest to newest, followed by the
initial state of the new execution. Infer the demonstrated strategy, predict
the complete subgoal plan for the new execution, then return its first
next_subgoal.
The overall image resolution is height={image_height}, width={image_width} pixels.

{observation_lines}

Do not treat actions in the demonstration as progress in the new execution.
Any target point must use original pixel coordinates from the top-left, not
normalized 0-1000 coordinates.
Start directly with the JSON object. Output JSON only.
"""


def build_system_prompt(
    *,
    subgoals: str,
    notes: str,
    example: str,
    include_description: bool,
    point_rules: str | None = None,
    initial_schema: str | None = None,
    followup_schema: str | None = None,
    description_rule_override: str | None = None,
) -> str:
    if include_description:
        description_rule = (
            "Include a non-empty description of no more than 50 English words and "
            "at most two short sentences. State only directly visible evidence about "
            "whether the current subgoal completed. Do not reconsider prior subgoals, "
            "plan future actions, list alternatives, or include step-by-step reasoning. "
            "Do not use phrases such as 'wait', 'actually', 'let me re-examine', "
            "'I need to reconsider', or 'looking more carefully'."
        )
        default_initial_schema = INITIAL_SCHEMA_WITH_DESCRIPTION
        default_followup_schema = FOLLOWUP_SCHEMA_WITH_DESCRIPTION
    else:
        description_rule = "Do not output a description field."
        default_initial_schema = INITIAL_SCHEMA_WITHOUT_DESCRIPTION
        default_followup_schema = FOLLOWUP_SCHEMA_WITHOUT_DESCRIPTION
    if description_rule_override is not None:
        description_rule = description_rule_override
    if initial_schema is None:
        initial_schema = default_initial_schema
    if followup_schema is None:
        followup_schema = default_followup_schema
    if point_rules is None:
        point_rules = (
            "5. For a pickup subgoal, output the target center in the latest "
            "overall image\n   as point=[y, x], using original pixel coordinates "
            "from the top-left.\n6. For a non-pickup subgoal and for STOP, "
            "output point as null."
        )
    return SYSTEM_TEMPLATE.format(
        subgoals=subgoals,
        notes=notes,
        example=example,
        point_rules=point_rules,
        description_rule=description_rule,
        initial_schema=initial_schema,
        followup_schema=followup_schema,
    )


@dataclass(frozen=True)
class TaskPromptSpec:
    task: str
    version: str
    include_description: bool
    system_prompt: str
    initial_user_template: str = INITIAL_USER_TEMPLATE
    followup_user_template: str = FOLLOWUP_USER_TEMPLATE
    include_wrist: bool = True
    initial_observation_mode: InitialObservationMode = "current"

    @property
    def content_hash(self) -> str:
        payload = {
            "task": self.task,
            "version": self.version,
            "include_description": self.include_description,
            "system_prompt": self.system_prompt,
            "initial_user_template": self.initial_user_template,
            "followup_user_template": self.followup_user_template,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def metadata(self) -> dict[str, str | bool]:
        return {
            "task": self.task,
            "version": self.version,
            "include_description": self.include_description,
            "include_wrist": self.include_wrist,
            "sha256": self.content_hash,
        }

    def build_messages(
        self,
        *,
        call_type: CallType,
        task_goal: str,
        history_subgoals: list[str] | None = None,
        current_subgoal: str | None = None,
        observation_count: int = 0,
        overall_image_shape: tuple[int, int] | None = None,
    ) -> list[dict[str, str]]:
        if overall_image_shape is None or len(overall_image_shape) < 2:
            raise ValueError("Calls requiring pixel grounding need overall_image_shape")
        image_height = int(overall_image_shape[0])
        image_width = int(overall_image_shape[1])
        if image_height < 1 or image_width < 1:
            raise ValueError("overall_image_shape must contain positive height/width")
        if call_type == "initial":
            if self.initial_observation_mode == "demonstration":
                if observation_count < 2:
                    raise ValueError(
                        "Demonstration initial calls require at least one demo "
                        "observation and one execution-state observation"
                    )
                if self.include_wrist:
                    observation_lines = "\n".join(
                        (
                            f"  Demonstration observation {index} overall: <image>\n"
                            f"  Demonstration observation {index} wrist: <image>"
                        )
                        for index in range(1, observation_count)
                    )
                    observation_lines += (
                        "\n  New-execution initial overall: <image>\n"
                        "  New-execution initial wrist: <image>"
                    )
                else:
                    observation_lines = "\n".join(
                        f"  Demonstration observation {index} overall: <image>"
                        for index in range(1, observation_count)
                    )
                    observation_lines += "\n  New-execution initial overall: <image>"
            else:
                if observation_count not in (0, 1):
                    raise ValueError(
                        "Current-state initial calls use one initial observation"
                    )
                observation_lines = ""
            user_prompt = self.initial_user_template.format(
                task_goal=task_goal,
                image_height=image_height,
                image_width=image_width,
                observation_lines=observation_lines,
            )
        else:
            if not current_subgoal:
                raise ValueError("Follow-up calls require current_subgoal")
            if observation_count < 1:
                raise ValueError("Follow-up calls require at least one recent observation")
            history = "; ".join(
                f"{index}. {subgoal}"
                for index, subgoal in enumerate(history_subgoals or [], start=1)
            ) or "none"
            if self.include_wrist:
                observation_lines = "\n".join(
                    (
                        f"  Observation {index} overall: <image>\n"
                        f"  Observation {index} wrist: <image>"
                    )
                    for index in range(1, observation_count + 1)
                )
            else:
                observation_lines = "\n".join(
                    f"  Observation {index} overall: <image>"
                    for index in range(1, observation_count + 1)
                )
            user_prompt = self.followup_user_template.format(
                task_goal=task_goal,
                history=history,
                current_subgoal=current_subgoal,
                observation_lines=observation_lines,
                image_height=image_height,
                image_width=image_width,
            )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
