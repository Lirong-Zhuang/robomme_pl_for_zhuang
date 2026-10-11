"""QwenVL Manager prompt with deterministic interaction verification."""

from mme_vla_suite.prompts.manager import qwenvl_v1


VERSION = "qwenvl_interaction_verify_v2"

SIMPLE_SYSTEM_PROMPT = qwenvl_v1.SIMPLE_SYSTEM_PROMPT
GROUNDED_SYSTEM_PROMPT = qwenvl_v1.GROUNDED_SYSTEM_PROMPT
INITIAL_USER_PROMPT_TEMPLATE = qwenvl_v1.INITIAL_USER_PROMPT_TEMPLATE
FOLLOWUP_USER_PROMPT_TEMPLATE = qwenvl_v1.FOLLOWUP_USER_PROMPT_TEMPLATE
REPORTER_COMPLETED_TEMPLATE = qwenvl_v1.REPORTER_COMPLETED_TEMPLATE
REPORTER_INCOMPLETE_TEMPLATE = qwenvl_v1.REPORTER_INCOMPLETE_TEMPLATE
REPORTER_MISSING_TEMPLATE = qwenvl_v1.REPORTER_MISSING_TEMPLATE

VERIFICATION_SYSTEM_PROMPT = (
    "You infer concrete interaction relations for one robot subgoal from the "
    "current observation."
)
VERIFICATION_USER_PROMPT_TEMPLATE = (
    "{video_prefix}Task: {task_goal}\n"
    "SUBGOAL TO VERIFY: {subgoal_to_verify}\n"
    "<image>Observe only this subgoal in the current image. Use only SUBGOAL TO VERIFY "
    "to choose the applicable interaction pairs. If SUBGOAL TO VERIFY contains 'pick "
    "up', output "
    "[[\"robot\",\"<relationship>\",\"<color> cube\"],[\"<color> cube\","
    "\"<relationship>\",\"table\"]]. If SUBGOAL "
    "TO VERIFY contains both 'put' and 'bin', the interaction pair is "
    "[[\"<color> cube\",\"<relationship>\",\"bin\"]]. If SUBGOAL TO VERIFY contains "
    "both 'press' "
    "and 'button', the interaction pair is [[\"robot\",\"<relationship>\","
    "\"button\"]]. Never mix interaction pairs from different cases. Use the Task and "
    "current image to resolve references. Replace <color> with the actual cube color. "
    "Replace each <relationship> with one concrete uppercase relation "
    "inferred from the current image, such as HOLDS, ABOVE, ON, IN, OUTSIDE, PRESSES, "
    "TOUCHES, NEAR, or NA. Use NA only when no concrete relation can be inferred. Never "
    "output <relationship> or RELATIONSHIP literally. Do not decide whether the subgoal "
    "is complete, do not output a Boolean, and do not choose the next subgoal. Output "
    "exactly one minified JSON line: "
    "{{\"interaction_pairs\":<inferred pairs>}}."
)
