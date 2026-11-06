"""StopCube-specific prompts and deterministic action mapping."""

from __future__ import annotations

import re

from .base import FOLLOWUP_USER_TEMPLATE
from .base import FOLLOWUP_USER_TEMPLATE_OVERALL_ONLY
from .base import INITIAL_USER_TEMPLATE
from .base import INITIAL_USER_TEMPLATE_OVERALL_ONLY
from .base import TaskPromptSpec
from .base import build_system_prompt


TASK_NAME = "StopCube"

NOTES = """1. Read the requested target-visit count N from the task goal. In the
initial subgoal_sequence, first move above the button, then include exactly one
numbered remain-static subgoal for every visit from the first through the
(N-1)th, followed by the numbered press subgoal for the Nth visit.
2. In the initial plan and throughout every follow-up call, always use the full
form "remain static while the cube reaches the target for the [ordinal] time".
Never output the bare or shortened form "remain static". Keep the ordinal in
current_subgoal, next_subgoal, and the resulting history consistent with the
number of distinct visits observed so far.
3. Estimate the cube's speed and count distinct target visits. The observation
clips are continuous: continuous frames from one target occupancy count as only
one visit. Do not advance the ordinal until a new visit is visibly observed.
Target visits observed while moving above the button to prepare still count.
When preparation completes, skip every numbered remain-static subgoal whose
visit has already occurred. If visits 1 through N-1 have already occurred and
the cube is approaching visit N, output the numbered press subgoal directly.
4. Each execution of a numbered remain-static subgoal advances to one incremental
absolute-timestep checkpoint. If its numbered visit has not yet been observed,
return completed=false and repeat the exact same full numbered next_subgoal.
After visit i is observed and another non-final waiting visit remains, return
completed=true and advance to the full numbered remain-static subgoal for i+1.
5. After the (N-1)th visit, keep the (N-1)th numbered remain-static subgoal active
while the cube travels toward its final visit. Do not output the press subgoal
merely because the (N-1)th visit completed. Only when the cube is approaching the
target for the Nth time, return completed=true and output the full numbered press
subgoal: "as the cube approaches the target for the [ordinal] time, press the
button".
6. The press motion takes time, so do not wait until the cube is already perfectly
centered on its final requested visit. Once the numbered press subgoal is active,
keep that exact subgoal until the press completes, then output STOP only if the
cube is stopped on the target.
7. The target is a purple disk.
8. All subgoals and STOP require point=null."""

SUBGOALS = """- move to the top of the button to prepare
- remain static while the cube reaches the target for the [first/second/third/...] time
- as the cube approaches the target for the [second/third/fourth/...] time, press the button
- STOP, only after the cube is stopped on the requested target visit"""

COMBINED_EXAMPLE = """If the task goal requests stopping on the third target visit, use exactly:
1. move to the top of the button to prepare
2. remain static while the cube reaches the target for the first time
3. remain static while the cube reaches the target for the second time
4. as the cube approaches the target for the third time, press the button

During follow-up calls, repeat the exact active numbered remain-static string
until its transition rule above is satisfied. Never shorten it to "remain static".
For example, if the first visit occurs while preparing for a two-visit task,
the first follow-up may go directly to "as the cube approaches the target for
the second time, press the button" instead of activating an obsolete first wait."""


SUBGOAL_ACTION_RULES: tuple[tuple[str, str], ...] = (
    (
        r"\b(?:move|go)\b.*\b(?:top|above|over)\b.*\bbutton\b.*\bprepare\b",
        "solve_button_ready",
    ),
    (r"\b(?:remain|stay|hold)\b.*\bstatic\b", "solve_hold_obj_absTimestep"),
    (r"\bpress\b.*\bbutton\b", "solve_button"),
)

POINT_ACTION_NAMES: tuple[str, ...] = ()


def _prompt(*, include_description: bool, include_wrist: bool) -> TaskPromptSpec:
    suffix = "with_description" if include_description else "without_description"
    view = "paired" if include_wrist else "overall_only"
    point_rules = (
        "5. Output point as null for every StopCube subgoal and for STOP.\n"
        "6. Never output pixel coordinates for this task."
    )
    return TaskPromptSpec(
        task=TASK_NAME,
        version=f"stop_cube_oracle_planner_v2_numbered_waits_{view}_{suffix}",
        include_description=include_description,
        include_wrist=include_wrist,
        system_prompt=build_system_prompt(
            subgoals=SUBGOALS,
            notes=NOTES,
            example=COMBINED_EXAMPLE,
            include_description=include_description,
            point_rules=point_rules,
        ),
        initial_user_template=(
            INITIAL_USER_TEMPLATE if include_wrist else INITIAL_USER_TEMPLATE_OVERALL_ONLY
        ),
        followup_user_template=(
            FOLLOWUP_USER_TEMPLATE
            if include_wrist
            else FOLLOWUP_USER_TEMPLATE_OVERALL_ONLY
        ),
    )


PROMPT_WITH_DESCRIPTION = _prompt(include_description=True, include_wrist=True)
PROMPT_WITHOUT_DESCRIPTION = _prompt(include_description=False, include_wrist=True)
PROMPT_OVERALL_ONLY_WITH_DESCRIPTION = _prompt(
    include_description=True, include_wrist=False
)
PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION = _prompt(
    include_description=False, include_wrist=False
)


def match_action_name(subgoal: str) -> str | None:
    text = " ".join(subgoal.casefold().replace("_", " ").split())
    for pattern, action_name in SUBGOAL_ACTION_RULES:
        if re.search(pattern, text):
            return action_name
    return None
