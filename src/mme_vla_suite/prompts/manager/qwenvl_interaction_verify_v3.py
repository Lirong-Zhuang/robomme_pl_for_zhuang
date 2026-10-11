"""QwenVL Manager prompt with a separate Reporter-success verification call."""

from mme_vla_suite.prompts.manager import qwenvl_v1


VERSION = "qwenvl_interaction_verify_v3"

SIMPLE_SYSTEM_PROMPT = qwenvl_v1.SIMPLE_SYSTEM_PROMPT
GROUNDED_SYSTEM_PROMPT = qwenvl_v1.GROUNDED_SYSTEM_PROMPT
INITIAL_USER_PROMPT_TEMPLATE = qwenvl_v1.INITIAL_USER_PROMPT_TEMPLATE
FOLLOWUP_USER_PROMPT_TEMPLATE = qwenvl_v1.FOLLOWUP_USER_PROMPT_TEMPLATE
REPORTER_COMPLETED_TEMPLATE = qwenvl_v1.REPORTER_COMPLETED_TEMPLATE
REPORTER_INCOMPLETE_TEMPLATE = qwenvl_v1.REPORTER_INCOMPLETE_TEMPLATE
REPORTER_MISSING_TEMPLATE = qwenvl_v1.REPORTER_MISSING_TEMPLATE

VERIFICATION_SYSTEM_PROMPT = (
    "You verify one robot subgoal from the current observation."
)
VERIFICATION_USER_PROMPT_TEMPLATE = (
    "{video_prefix}Task: {task_goal}\n"
    "SUBGOAL TO VERIFY: {subgoal_to_verify}\n"
    "<image>First, use SUBGOAL TO VERIFY and the current image to infer the relevant "
    "interaction pairs. Write each pair as "
    "[\"subject\",\"<relationship>\",\"object\"], replacing <relationship> with the "
    "concrete uppercase relation inferred from the image. Then decide whether SUBGOAL "
    "TO VERIFY is complete using only the inferred pairs and relations. Output exactly "
    "one minified JSON line: "
    "{{\"interaction_pairs\":<inferred pairs>,"
    "\"verification_passed\":<true_or_false>}}."
)
