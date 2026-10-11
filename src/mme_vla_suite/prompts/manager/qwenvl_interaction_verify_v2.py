"""QwenVL Manager prompt with a separate Reporter-success verification call."""

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
    "You verify whether one robot subgoal is complete from the current observation."
)
VERIFICATION_USER_PROMPT_TEMPLATE = (
    "{video_prefix}Task: {task_goal}\n"
    "SUBGOAL TO VERIFY: {subgoal_to_verify}\n"
    "<image>Verify only this subgoal from the current observation. "
    "If SUBGOAL TO VERIFY contains 'pick up', the interaction pairs are "
    "[[\"robot\",\"<relationship>\",\"<color> cube\"],[\"<color> cube\","
    "\"<relationship>\",\"table\"]], and the completion condition is [[\"robot\","
    "\"HOLDS\",\"<color> cube\"],[\"<color> cube\",\"ABOVE\",\"table\"]]. If SUBGOAL "
    "TO VERIFY contains both 'put' and 'bin', the interaction pair is "
    "[[\"<color> cube\",\"<relationship>\",\"bin\"]], and the completion condition is "
    "[[\"<color> cube\",\"IN\",\"bin\"]]. If SUBGOAL TO VERIFY contains both 'press' "
    "and 'button', the interaction pair is [[\"robot\",\"<relationship>\","
    "\"button\"]], and the completion condition is [[\"robot\",\"PRESSES\","
    "\"button\"]]. Never mix interaction pairs from different cases. Use the Task and "
    "current image to resolve references. Replace <color> with the "
    "actual cube color. Replace each <relationship> with one concrete uppercase relation "
    "inferred from the current image, such as HOLDS, ABOVE, ON, IN, OUTSIDE, PRESSES, "
    "TOUCHES, NEAR, or NA. Use NA only when no concrete relation can be inferred. Set "
    "verification_passed=true only if the inferred interaction pairs exactly match the "
    "applicable completion condition after replacing <color>. If any relationship is NA "
    "or any pair differs, set verification_passed=false. Output only this verification "
    "signal as one minified JSON line: "
    "{{\"verification_passed\":<true_or_false>}}."
)
