"""Reporter prompt that also emits subgoal-specific interaction pairs.

This keeps the temporal completion judgement from ``interaction_aware_v2``
and moves the interaction-pair observation task from the Manager to the
Reporter. Evaluation currently consumes only ``success``; the pairs are
recorded in the raw Reporter response for inspection.
"""

VERSION = "interaction_pairs_v3"

SYSTEM_PROMPT = (
    "You are a helpful assistant to determine whether the current robot subgoal "
    "is complete by comparing observation before executing the current subgoal "
    "with a recent sequence of observations, and to report exactly one interaction "
    "relation relevant to that subgoal. Begin your response immediately with { "
    "and emit exactly one JSON object on exactly one line in this form: "
    '{"success":<true_or_false>,"interaction_pairs":<inferred pairs>}. '
    "Stop immediately after the final }. Do not emit a newline, a second JSON "
    "object, a Markdown code fence, commentary, or an explanation."
)

USER_PROMPT_TEMPLATE = (
    "Current Subgoal: {subgoal}\n"
    "Observation before executing the current subgoal: <image>\n"
    "Recent observations after execution, from oldest to newest:\n"
    "{observation_lines}\n"
    "Determine whether the current subgoal is complete by comparing the "
    "pre-execution observation with the recent observation sequence. Use the "
    "current subgoal to decide which state evidence is relevant. For robot-only "
    "subgoals, the robot's state or pose may be sufficient. For subgoals involving "
    "an object or the environment, verify the corresponding state change instead "
    "of assuming success merely because the robot's motion has stopped.\n"
    "Also output exactly one interaction pair applicable to Current Subgoal. "
    "interaction_pairs must always be a list containing exactly one triplet of "
    "the form [[\"<subject>\",\"<relationship>\",\"<object>\"]]; never output "
    "zero pairs or more than one pair. If Current Subgoal contains 'pick up', "
    "output [[\"robot\",\"<relationship>\",\"<color> cube\"]]. If it contains "
    "both 'put' and 'bin', output [[\"<color> cube\",\"<relationship>\","
    "\"bin\"]]. If it contains both 'press' and 'button', output "
    "[[\"robot\",\"<relationship>\",\"button\"]]. For any other subgoal, choose "
    "the single subject-object pair most directly described by Current Subgoal; "
    "when only one non-robot entity is named, use robot as the subject. Never mix "
    "pairs from different cases. Replace <color> with the actual cube color. "
    "Infer the entities and <relationship> only from the latest image in Recent "
    "observations, explicitly labelled '(current observation)'. "
    "Use one concrete uppercase relation such as HOLDS, ABOVE, ON, IN, OUTSIDE, "
    "PRESSES, TOUCHES, NEAR, or NA. For a pick-up pair [robot, relationship, "
    "cube], use HOLDS only when the current observation visibly shows the gripper "
    "closed around and supporting the cube, with the cube lifted from its support. "
    "Use ABOVE when the robot or gripper is spatially above the cube with a visible "
    "air gap but is not gripping or supporting it. Thus ABOVE is not HOLDS. If the "
    "gripper merely contacts the cube without lifting and supporting it, use "
    "TOUCHES; mere proximity is NEAR. Use ON when the subject is visibly touching "
    "and supported by the object. Use NA when the relation cannot be determined "
    "from the current observation. Never output <relationship> or RELATIONSHIP "
    "literally. Always "
    "include interaction_pairs whether success is true or false. Return exactly "
    "one single-line JSON object and nothing else."
)
