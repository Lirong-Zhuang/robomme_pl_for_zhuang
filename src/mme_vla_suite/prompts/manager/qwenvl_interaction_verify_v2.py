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
    "The Reporter reports that the last {subgoal_name} is complete. The Manager must: "
    "(1) Classify the current subgoal as pick-up, put-into-bin, press-button, or other. "
    "Classification alone does not prove completion. (2) Construct and complete the matching "
    "interaction pairs: pick-up => "
    "[[\"robot\",\"RELATIONSHIP\",\"<color> cube\"],[\"<color> cube\",\"RELATIONSHIP\",\"table\"]]; "
    "put-into-bin => [[\"<color> cube\",\"RELATIONSHIP\",\"bin\"]]; press-button => "
    "[[\"robot\",\"RELATIONSHIP\",\"button\"]]; other => infer the minimum concrete "
    "pairs. Resolve references from history, replace <color> with the actual cube color, and "
    "infer each RELATIONSHIP from the current image. Always output the inferred pairs, even "
    "when verification fails. "
    "(3) Compare the inferred pairs with these completion conditions: pick-up => "
    "[[\"robot\",\"HOLDS\",\"<color> cube\"],[\"<color> cube\",\"ABOVE\",\"table\"]]; "
    "put-into-bin => [[\"<color> cube\",\"IN\",\"bin\"]]; press-button => "
    "[[\"robot\",\"PRESSES\",\"button\"]]; other => all inferred goal relations are "
    "satisfied. Verification succeeds only if every required pair matches its completion "
    "condition. If successful, set verification_passed=true and "
    "choose the next {subgoal_name}; otherwise set it to false and keep the same "
    "{subgoal_name}. Output exactly one minified JSON line: "
    "{{\"interaction_pairs\":<matching pairs>,\"verification_passed\":"
    "<true_or_false>,\"subgoal\":\"<selected subgoal>\"}}."
)

REPORTER_INCOMPLETE_TEMPLATE = (
    "The Reporter determined that the last {subgoal_name} is incomplete. "
    "Output that same {subgoal_name}."
)
REPORTER_MISSING_TEMPLATE = (
    "The Reporter provided no result for the last {subgoal_name}. "
    "Output that same {subgoal_name}."
)
