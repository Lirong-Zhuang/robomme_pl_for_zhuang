"""Prompts shared by Reporter training and inference."""

REPORTER_SYSTEM_PROMPT = (
    "You are a helpful assistant to determine whether the current robot subgoal "
    "is complete by comparing two observations. "
    'Return only {"success": true} or {"success": false}. '
)

REPORTER_USER_PROMPT = (
    "Current Subgoal: {subgoal}\n"
    "Observation before executing the current subgoal: <image>\n"
    "Observation after execution: <image>\n"
    "Determine whether the current subgoal is complete based on the observations before and after execution. "
)


def format_reporter_user_prompt(subgoal: str) -> str:
    """Insert a subgoal into the exact user prompt used at inference time."""
    return REPORTER_USER_PROMPT.replace("{subgoal}", subgoal)
