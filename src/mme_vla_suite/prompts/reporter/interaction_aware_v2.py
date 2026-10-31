"""Interaction-aware Reporter prompt introduced by Trinity v2.2."""

VERSION = "interaction_aware_v2"

SYSTEM_PROMPT = (
    "You are a helpful assistant to determine whether the current robot subgoal "
    "is complete by comparing observation before executing the current subgoal with a recent sequence "
    "of observations. "
    'Return only {"success": true} or {"success": false}. '
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
    "of assuming success merely because the robot's motion has stopped. "
)
