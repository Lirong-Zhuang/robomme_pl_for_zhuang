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
    "{video_prefix}The task goal is: {task_goal}\n"
    "The history of previous predicted {subgoal_name_plural} are: {history}\n"
    "{reporter_text}\n"
    "<image>What's the next {subgoal_name} based on current observation and the result from the Reporter? "
    "If the Reporter determines that the last subgoal is not complete, output the same subgoal."
)
REPORTER_COMPLETED_FOLLOWUP_USER_PROMPT_TEMPLATE = (
    "{video_prefix}The task goal is: {task_goal}\n"
    "The history of previous predicted {subgoal_name_plural} are: {history}\n"
    "{reporter_text}\n"
    "SUBGOAL TO VERIFY: {subgoal_to_verify}\n"
    "<image>What's the next {subgoal_name} based on current observation and the result from the Reporter? "
    "If the Reporter determines that the last subgoal is not complete, output the same subgoal."
)

REPORTER_COMPLETED_TEMPLATE = (
    "The Reporter reports that SUBGOAL TO VERIFY is complete. Verify it from the current "
    "image. Use only SUBGOAL TO VERIFY to determine the interaction pairs. If SUBGOAL TO "
    "VERIFY contains 'pick up', the interaction pairs are "
    "[[\"robot\",\"<relationship>\",\"<color> cube\"],[\"<color> cube\","
    "\"<relationship>\",\"table\"]], and the completion condition is [[\"robot\","
    "\"HOLDS\",\"<color> cube\"],[\"<color> cube\",\"ABOVE\",\"table\"]]. If SUBGOAL "
    "TO VERIFY contains both 'put' and 'bin', the interaction pair is "
    "[[\"<color> cube\",\"<relationship>\",\"bin\"]], and the completion condition is "
    "[[\"<color> cube\",\"IN\",\"bin\"]]. If SUBGOAL TO VERIFY contains both 'press' "
    "and 'button', the interaction pair is [[\"robot\",\"<relationship>\","
    "\"button\"]], and the completion condition is [[\"robot\",\"PRESSES\","
    "\"button\"]]. Never mix interaction pairs from different cases. Use the full history "
    "to resolve references and choose the next unfinished action, but never use another "
    "history item to determine the pairs for SUBGOAL TO VERIFY. Replace <color> with the "
    "actual cube color. Replace each <relationship> with one concrete uppercase relation "
    "inferred from the current image, such as HOLDS, ABOVE, ON, IN, OUTSIDE, PRESSES, "
    "TOUCHES, NEAR, or NA. Use NA only when no concrete relation can be inferred. Never "
    "output <relationship> or RELATIONSHIP literally. Always output the inferred pairs, "
    "even when verification fails. Set verification_passed=true only if the final "
    "interaction_pairs exactly match the applicable completion condition after replacing "
    "<color>. If <relationship> or RELATIONSHIP remains, or any pair differs, set "
    "verification_passed=false "
    "and output SUBGOAL TO VERIFY unchanged. If verification succeeds, use the Task and "
    "full history to output the next unfinished concrete {subgoal_name}; do not skip "
    "remaining repeated actions. Output "
    "exactly one minified JSON line: {{\"interaction_pairs\":<inferred pairs>,"
    "\"verification_passed\":<true_or_false>,\"subgoal\":\"<concrete action>\"}}."
)

REPORTER_INCOMPLETE_TEMPLATE = (
    "The Reporter determined that the last {subgoal_name} is incomplete. "
    "Output that same {subgoal_name}."
)
REPORTER_MISSING_TEMPLATE = (
    "The Reporter provided no result for the last {subgoal_name}. "
    "Output that same {subgoal_name}."
)
