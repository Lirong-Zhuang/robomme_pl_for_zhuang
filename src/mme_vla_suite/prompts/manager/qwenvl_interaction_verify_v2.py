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
    qwenvl_v1.FOLLOWUP_USER_PROMPT_TEMPLATE
    + " If the Reporter determines that the last subgoal is complete, verify it again "
    "from the current observation before outputting the next subgoal; if verification "
    "fails, output the same subgoal."
)

REPORTER_COMPLETED_TEMPLATE = (
    "The Reporter determined that the last predicted {subgoal_name} has been completed. "
    "Only for this completed result, verify the last subgoal from the current observation "
    "before changing to the next subgoal. Represent each interaction pair as one compact "
    "three-string JSON array: [\"first object\",\"RELATIONSHIP\",\"second object\"]. "
    "It means that the first object is in the stated RELATIONSHIP to the second object. "
    "Keep RELATIONSHIP uppercase. Select the concrete objects, including their colors when "
    "present, from the last subgoal and use the subgoal history to resolve references such "
    "as 'it'. Check only the relevant interaction pair or pairs. Apply these "
    "requirements: "
    "(1) For a pick-up subgoal involving a colored cube, interaction_pairs must always contain "
    "exactly [[\"robot\",\"HOLDS\",\"<color> cube\"],"
    "[\"<color> cube\",\"ABOVE\",\"table\"]]. Verification passes only if both "
    "relations are clearly true. "
    "Here, HOLDS means that the cube is secured by the gripper, and ABOVE means that the "
    "cube is visibly lifted clear of and no longer supported by the table. Robot proximity, "
    "touching the cube, or a closed gripper alone is not enough. "
    "(2) For a put-into-bin subgoal, interaction_pairs must always contain exactly "
    "[[\"<target block or cube>\",\"IN\",\"<target bin>\"]]. The relation is true only "
    "when the target object is inside the intended bin, not merely above it or touching its rim. "
    "(3) For a button subgoal, interaction_pairs must always contain exactly "
    "[[\"robot\",\"PRESSES\",\"<target button>\"]]. The relation is true only when the "
    "robot is in pressing contact with the intended button and the button appears depressed "
    "or activated; being near or merely touching the button is not enough. "
    "Use the original Reporter-based behavior for subgoals outside these three classes. "
    "Always include every required interaction pair for the applicable class, even when the "
    "visual check fails; never replace a required list with an empty list. Set "
    "verification_passed to true only if every listed relation is clearly true in the current "
    "image. If it is true, select the next subgoal. Otherwise, set it to false and select the "
    "same last subgoal. Return only one compact JSON object with this schema: "
    "{{\"interaction_pairs\":[[\"robot\",\"HOLDS\",\"green cube\"],"
    "[\"green cube\",\"ABOVE\",\"table\"]],\"verification_passed\":true,"
    "\"subgoal\":\"the selected {subgoal_name}\"}}. "
    "Use an empty interaction_pairs list only for subgoals outside the three classes, and "
    "output no text outside the JSON object."
)

REPORTER_INCOMPLETE_TEMPLATE = qwenvl_v1.REPORTER_INCOMPLETE_TEMPLATE
REPORTER_MISSING_TEMPLATE = qwenvl_v1.REPORTER_MISSING_TEMPLATE
