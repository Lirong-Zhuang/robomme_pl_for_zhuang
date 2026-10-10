"""QwenVL Manager prompt with verification gated by Reporter success.

The system and initial prompts are inherited from ``qwenvl_v1``. The follow-up
prompt adds one short success-verification gate, while the detailed interaction
requirements are injected only when the Reporter returns ``True``.
"""

from mme_vla_suite.prompts.manager import qwenvl_v1


VERSION = "qwenvl_interaction_verify_v2"

SIMPLE_SYSTEM_PROMPT = qwenvl_v1.SIMPLE_SYSTEM_PROMPT
GROUNDED_SYSTEM_PROMPT = qwenvl_v1.GROUNDED_SYSTEM_PROMPT
INITIAL_USER_PROMPT_TEMPLATE = qwenvl_v1.INITIAL_USER_PROMPT_TEMPLATE
FOLLOWUP_USER_PROMPT_TEMPLATE = (
    "{video_prefix}Task: {task_goal}\n"
    "{subgoal_name_plural}: {history}\n"
    "{reporter_text}\n"
    "Current image: <image>"
)

REPORTER_COMPLETED_TEMPLATE = (
    "The Reporter says the last {subgoal_name} is complete; verify it yourself in the current "
    "image. Resolve 'it' and targets from history. Required pairs and pass conditions: "
    "(1) Pick up: [[\"robot\",\"HOLDS\",\"<target>\"],"
    "[\"<target>\",\"ABOVE\",\"table\"]]. Pass only if the gripper visibly holds the "
    "target and it is clear of the table. "
    "(2) Put into bin: [[\"<target>\",\"IN\",\"<bin>\"]]. Pass only if the target is "
    "visibly inside the intended bin and released; above, on the rim, outside, or still held "
    "fails. (3) Press button: [[\"robot\",\"PRESSES\",\"<button>\"]]. Pass only if the "
    "robot visibly presses the intended button and it appears depressed or activated. "
    "Replace placeholders with concrete colored names. Pairs are checks: include them even "
    "when false or unclear, and never use [] for these tasks. For another task, infer minimal "
    "pairs. If all required relations are visible, set verification_passed=true and choose the "
    "next {subgoal_name}; otherwise set it false and keep the last one. Output exactly one "
    "minified JSON line in this key order: interaction_pairs, verification_passed, subgoal. "
    "No line breaks, markdown, or other text."
)

REPORTER_INCOMPLETE_TEMPLATE = (
    "The Reporter determined that the last {subgoal_name} is incomplete. "
    "Output that same {subgoal_name}."
)
REPORTER_MISSING_TEMPLATE = (
    "The Reporter provided no result for the last {subgoal_name}. "
    "Output that same {subgoal_name}."
)
