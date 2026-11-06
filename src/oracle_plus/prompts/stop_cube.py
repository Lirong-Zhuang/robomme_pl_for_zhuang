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
(N-1)th, followed by the immediate numbered press subgoal for the Nth visit.
2. In the initial plan and throughout every follow-up call, always use the full
form "remain static while the cube reaches the target for the [ordinal] time".
Never output the bare or shortened form "remain static". Keep the ordinal in
current_subgoal, next_subgoal, and the resulting history consistent with the
number of distinct visits observed so far.
3. Estimate the cube's speed and count distinct target visits. The observation
clips are continuous: continuous frames from one target occupancy count as only
one visit. Do not advance the ordinal until a new visit is visibly observed.
The subgoal-init observation is the boundary left by the preceding subgoal. Any
target occupancy already visible there has already been consumed and must not
complete the new subgoal. If the cube is on the target in that image, require it
to visibly leave and then re-enter before counting another visit. A continuous
occupancy spanning subgoal-init and recent observations is still only one visit.
Target visits observed while moving above the button to prepare still count.
When preparation completes, skip every numbered remain-static subgoal whose
visit has already occurred. If visits 1 through N-1 have already occurred and
the newest observations show that the cube is currently approaching visit N,
output the immediate numbered press subgoal directly.
4. Each execution of a numbered remain-static subgoal advances to one incremental
absolute-timestep checkpoint. If its numbered visit has not yet been observed,
return completed=false and repeat the exact same full numbered next_subgoal.
After visit i is observed and another non-final waiting visit remains, return
completed=true and advance to the full numbered remain-static subgoal for i+1.
5. After the (N-1)th visit, keep the (N-1)th numbered remain-static subgoal active
while the cube travels toward its final visit. Do not output the press subgoal
merely because the (N-1)th visit completed. Only when the cube is approaching the
target for the Nth time, return completed=true and output the full numbered press
subgoal: "press the button now because the cube is currently approaching the
target for the [ordinal] time". This is an immediate action statement, not an
instruction to press now merely because an earlier visit has finished.
6. The press motion takes time. After exactly N-1 distinct visits are confirmed,
use the direction and speed across the newest observations to predict the Nth
approach. Output the immediate press subgoal while the cube is visibly moving
toward the target but is still short of its center, early enough for the press
motion to finish as the cube reaches the target. Do not wait for overlap with the
target, and do not press while the cube is moving away after visit N-1.
7. The ordinal in the press subgoal must exactly equal N from the task goal. For
a fourth-visit goal, the third approach is non-final and must not trigger a press;
for a fifth-visit goal, neither the third nor fourth approach may trigger a press.
Once the numbered press subgoal is active, keep that exact subgoal until the press
completes, then output STOP only if the cube is stopped on the target.
8. The target is a purple disk.
9. All subgoals and STOP require point=null."""

SUBGOALS = """- move to the top of the button to prepare
- remain static while the cube reaches the target for the [first/second/third/...] time
- press the button now because the cube is currently approaching the target for
  the [second/third/fourth/...] time
- STOP, only after the cube is stopped on the requested target visit"""

COMBINED_EXAMPLE = """Let N be the exact requested target-visit ordinal. Use exactly:
1. move to the top of the button to prepare
2 through N. one numbered remain-static subgoal for each visit from 1 through N-1
N+1. press the button now because the cube is currently approaching the target
for the Nth time

During follow-up calls, repeat the exact active numbered remain-static string
until its transition rule above is satisfied. Never shorten it to "remain static".
For example, if the first visit occurs while preparing for a two-visit task and
the cube is now moving away, do not output the press subgoal. Keep a numbered
remain-static subgoal active until the newest observations show the cube is
currently approaching its second visit. Only then output "press the button now
because the cube is currently approaching the target for the second time"."""


SUBGOAL_ACTION_RULES: tuple[tuple[str, str], ...] = (
    (
        r"\b(?:move|go)\b.*\b(?:top|above|over)\b.*\bbutton\b.*\bprepare\b",
        "solve_button_ready",
    ),
    (r"\b(?:remain|stay|hold)\b.*\bstatic\b", "solve_hold_obj_absTimestep"),
    (
        r"\bpress\b.*\bbutton\b.*\bnow\b.*\bcurrently\b.*\bapproach(?:ing)?\b",
        "solve_button",
    ),
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
        version=f"stop_cube_oracle_planner_v4_vlm_timed_press_{view}_{suffix}",
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
