"""MoveCube-specific demonstration prompt and deterministic action mapping."""

from __future__ import annotations

import re

from .base import FOLLOWUP_USER_TEMPLATE
from .base import FOLLOWUP_USER_TEMPLATE_OVERALL_ONLY
from .base import INITIAL_DEMONSTRATION_USER_TEMPLATE
from .base import INITIAL_DEMONSTRATION_USER_TEMPLATE_OVERALL_ONLY
from .base import TaskPromptSpec
from .base import build_system_prompt


TASK_NAME = "MoveCube"

NOTES = """1. The purple disk is the target.
2. Infer exactly which of three strategies the pre-recorded demonstration uses:
(A) close the gripper and push the cube directly; (B) pick up the cube and place
it on the target; or (C) pick up the peg and hook the cube to the target.
3. Reproduce only the demonstrated strategy in the new execution. Do not mix
actions from different strategies.
4. If the active subgoal is not visibly complete, keep it active.
5. As in the main oracle-planner prompt, every MoveCube action uses point=null;
the task has fixed cube, target, and peg objects."""

SUBGOALS = """- close gripper and push the cube to the target
- pick up the cube
- place the cube onto the target
- pick up the peg
- hook the cube to the target with the peg
- STOP, only after the demonstrated strategy has completed"""

COMBINED_EXAMPLE = """Choose exactly one demonstrated strategy:
A) push: close gripper and push the cube to the target
B) pick-and-place: pick up the cube; place the cube onto the target
C) peg hook: pick up the peg; hook the cube to the target with the peg"""

POINT_RULES = """5. For every MoveCube subgoal and for STOP, output point as null.
6. Never output point=[y, x] for this task."""


SUBGOAL_ACTION_RULES: tuple[tuple[str, str], ...] = (
    (r"\bhook\b.*\bcube\b.*\btarget\b.*\bpeg\b", "solve_push_to_target_with_peg"),
    (r"\b(?:close\b.*\bgripper\b.*)?\bpush\b.*\bcube\b.*\btarget\b", "solve_push_to_target"),
    (r"\bpick\s+up\b.*\bcube\b", "solve_pickup"),
    (r"\b(?:place|put)\b.*\bcube\b.*\b(?:onto|on)\b.*\btarget\b", "solve_putonto_whenhold"),
    (r"\bpick\s+up\b.*\bpeg\b", "grasp_and_lift_peg_side"),
)

POINT_ACTION_NAMES: tuple[str, ...] = ()


def _prompt(*, include_description: bool, include_wrist: bool) -> TaskPromptSpec:
    suffix = "with_description" if include_description else "without_description"
    view = "paired" if include_wrist else "overall_only"
    return TaskPromptSpec(
        task=TASK_NAME,
        version=f"move_cube_oracle_planner_demo_v1_{view}_{suffix}",
        include_description=include_description,
        include_wrist=include_wrist,
        initial_observation_mode="demonstration",
        system_prompt=build_system_prompt(
            subgoals=SUBGOALS,
            notes=NOTES,
            example=COMBINED_EXAMPLE,
            include_description=include_description,
            point_rules=POINT_RULES,
        ),
        initial_user_template=(
            INITIAL_DEMONSTRATION_USER_TEMPLATE
            if include_wrist
            else INITIAL_DEMONSTRATION_USER_TEMPLATE_OVERALL_ONLY
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
