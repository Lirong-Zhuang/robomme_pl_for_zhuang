"""Original temporal Reporter prompt used by Trinity v2.1."""

VERSION = "temporal_v1"

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
    "Determine whether the current subgoal is complete from the observation "
    "before executing the current subgoal and the recent observation sequence. "
)
