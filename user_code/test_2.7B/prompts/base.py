"""Shared ordinary (non-grounded) prompts for task-specific 2.7B tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal


CallType = Literal["initial", "followup"]


SYSTEM_TEMPLATE = """You are a helpful assistant that guides a robot by predicting
the next ordinary language subgoal and selecting its matching high-level MPLib
wrapper action.

Possible subgoals:
{subgoals}

Possible actions:
{actions}

Task-specific notes:
{notes}

Example:
{example}

Consistency rules:
1. action_name must be the exact possible-action name corresponding to next_subgoal.
2. If the current subgoal is incomplete, keep both the current subgoal and its action.
3. If it is complete, advance both outputs to the next stage.
4. Only when the entire task is complete, output next_subgoal as STOP and action_name as null.

Output format:
For the initial call, return only:
```json
{{
  "subgoal_sequence": ["ordered subgoal 1", "ordered subgoal 2"],
  "action_sequence": ["matching wrapper 1", "matching wrapper 2"],
  "completed": false,
  "next_subgoal": "the first subgoal",
  "action_name": "its exact matching wrapper name"
}}
```

For a follow-up call, return only:
```json
{{
  "description": "briefly describe what the robot has or has not completed",
  "completed": true or false,
  "next_subgoal": "the next subgoal, or the current one if it is incomplete",
  "action_name": "the exact wrapper matching next_subgoal, or null for STOP"
}}
```
Return JSON only. Do not output analysis, reasoning, or think tags. Never invent
a subgoal or wrapper outside the supplied lists.
"""


INITIAL_USER_TEMPLATE = """The task goal is: {task_goal}
This is the initial call. Observe the initial state, predict the complete subgoal/action plan,
then return its first next_subgoal and matching action_name.
Initial overall image: <image>
Initial wrist image: <image>
Output JSON only.
"""


FOLLOWUP_USER_TEMPLATE = """The task goal is: {task_goal}
The history of previously active subgoals is: {history}
The current subgoal is: {current_subgoal}

Observation before executing the current subgoal:
  Overall image: <image>
  Wrist image: <image>
Recent paired observations after execution, oldest to newest:
{observation_lines}

Compare the initial pair with the recent temporal observations. Do not infer success merely because
the robot stopped moving. Decide whether the current subgoal is complete, then return the subgoal
that should be active now and its matching action. If incomplete, keep both current outputs.
Output JSON only.
"""


@dataclass(frozen=True)
class TaskPromptSpec:
    task: str
    version: str
    system_prompt: str
    initial_user_template: str = INITIAL_USER_TEMPLATE
    followup_user_template: str = FOLLOWUP_USER_TEMPLATE

    @property
    def content_hash(self) -> str:
        payload = {
            "task": self.task,
            "version": self.version,
            "system_prompt": self.system_prompt,
            "initial_user_template": self.initial_user_template,
            "followup_user_template": self.followup_user_template,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def metadata(self) -> dict[str, str]:
        return {
            "task": self.task,
            "version": self.version,
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
    ) -> list[dict[str, str]]:
        if call_type == "initial":
            if observation_count != 0:
                raise ValueError("Initial calls use only one initial paired observation")
            user_prompt = self.initial_user_template.format(
                task_goal=task_goal,
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
            observation_lines = "\n".join(
                (
                    f"  Observation {index} overall: <image>\n"
                    f"  Observation {index} wrist: <image>"
                )
                for index in range(1, observation_count + 1)
            )
            user_prompt = self.followup_user_template.format(
                task_goal=task_goal,
                history=history,
                current_subgoal=current_subgoal,
                observation_lines=observation_lines,
            )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

