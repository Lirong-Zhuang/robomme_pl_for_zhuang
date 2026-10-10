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
    "Independently verify that last subgoal from the current image before changing to the "
    "next subgoal. You, the Manager, must infer all of the following yourself: which "
    "interaction pair or pairs are relevant to the last subgoal, whether those relations "
    "are visually true, whether the last subgoal is complete, and which subgoal should run "
    "next. Do not treat the Reporter's completed result as visual proof. "
    "Use the last subgoal in the history as the task being verified. Resolve references such "
    "as 'it' from the preceding subgoals. Select concrete objects and include colors when "
    "present. Represent every inferred interaction pair as a compact three-string JSON array "
    "[\"first object\",\"RELATIONSHIP\",\"second object\"], meaning that the first object "
    "is in the stated relationship to the second object. Keep RELATIONSHIP uppercase. "
    "Apply these visual success standards: "
    "(1) Pick-up: infer the target cube or block from the subgoal. Success requires the "
    "intended target to be visibly secured by the robot gripper and visibly lifted clear of "
    "the table, with no support from the table. The relevant relations are robot HOLDS target "
    "and target ABOVE table, and both must be visibly true. Proximity, touching, or a closed "
    "gripper without a secured and lifted target is failure. "
    "(2) Put into bin: infer the target cube or block and the intended bin from the subgoal "
    "and history. Success requires the intended target to be visibly inside the intended bin "
    "and released by the robot. A target above the bin, touching or balanced on the rim, "
    "outside the bin, or still held by the gripper is failure. The relevant relation is target "
    "IN intended bin. "
    "(3) Press button: infer the intended button from the subgoal and history. Success requires "
    "the robot to be visibly pressing the intended button and the button to appear depressed "
    "or activated. Merely approaching or touching it without a visible press is failure. The "
    "relevant relation is robot PRESSES intended button. "
    "(4) Any other subgoal: infer the smallest set of concrete interaction pairs needed to "
    "verify that specific subgoal and judge them from the current image. "
    "Do not copy interaction pairs belonging to a different task class. Do not reuse a prior "
    "pair unless it is genuinely the relation required by the last subgoal. If any required "
    "relation is unclear or not visible, set verification_passed to false. Always list the "
    "interaction pairs you actually checked, even when verification fails. Set "
    "verification_passed to true only when every required relation and the full success "
    "standard for the last subgoal are visibly satisfied. When true, select the next "
    "{subgoal_name}; when false, select the same last {subgoal_name}. Return only one compact "
    "JSON object with exactly these keys: interaction_pairs, verification_passed, and subgoal. "
    "The interaction_pairs value must be a JSON list of the concrete triples you inferred, "
    "verification_passed must be true or false, and subgoal must be the selected "
    "{subgoal_name}. Output no text outside the JSON object."
)

REPORTER_INCOMPLETE_TEMPLATE = qwenvl_v1.REPORTER_INCOMPLETE_TEMPLATE
REPORTER_MISSING_TEMPLATE = qwenvl_v1.REPORTER_MISSING_TEMPLATE
