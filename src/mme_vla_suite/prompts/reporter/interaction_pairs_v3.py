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
    "with a recent sequence of observations, and to report the interaction "
    "relations relevant to that subgoal. Return only one minified JSON line in "
    "this exact form: "
    '{"success":<true_or_false>,"interaction_pairs":<inferred pairs>}. '
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
    "Also infer only the interaction pairs applicable to Current Subgoal. If it "
    "contains 'pick up', output [[\"robot\",\"<relationship>\",\"<color> "
    "cube\"],[\"<color> cube\",\"<relationship>\",\"table\"]]. If it contains "
    "both 'put' and 'bin', output [[\"<color> cube\",\"<relationship>\","
    "\"bin\"]]. If it contains both 'press' and 'button', output "
    "[[\"robot\",\"<relationship>\",\"button\"]]. For any other subgoal, output "
    "an empty interaction_pairs list. Never mix pairs from different cases. "
    "Replace <color> with the actual cube color. Infer each <relationship> from "
    "the current observation, using the recent sequence only to disambiguate the "
    "visible state change. Use one concrete uppercase relation such as HOLDS, "
    "ABOVE, ON, IN, OUTSIDE, PRESSES, TOUCHES, NEAR, or NA. Use ON when the cube "
    "is visibly touching and supported by the table. Use ABOVE when there is a "
    "visible air gap between the cube and the table. Use HOLDS only when the "
    "gripper is visibly gripping the cube and the cube is lifted from the table. "
    "Mere proximity is NEAR, not HOLDS. Use NA when the relation cannot be "
    "determined. Never output <relationship> or RELATIONSHIP literally. Always "
    "include interaction_pairs whether success is true or false."
)
