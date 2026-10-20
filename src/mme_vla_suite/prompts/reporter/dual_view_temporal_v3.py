"""Paired overall/wrist-view Reporter prompt introduced by Trinity v2.2."""

VERSION = "dual_view_temporal_v3"

SYSTEM_PROMPT = (
    "You are a helpful assistant that determines whether the current robot "
    "subgoal is complete from a recent temporal sequence of paired overall-camera "
    "and wrist-camera observations. "
    'Return only {"success": true} or {"success": false}. '
)

USER_PROMPT_TEMPLATE = (
    "Current Subgoal: {subgoal}\n"
    "Observation before executing the current subgoal:\n"
    "  Overall image: <image>\n"
    "  Wrist image: <image>\n"
    "Recent paired observations after execution, from oldest to newest:\n"
    "{observation_lines}\n"
    "Determine whether the current subgoal is complete by comparing the paired "
    "init observation with the {observation_count} paired recent observations. "
    "The overall "
    "view shows the scene-level state; "
    "the wrist view provides close-range evidence about the gripper and its "
    "interaction with objects. Use the current subgoal to decide which evidence "
    "is relevant. For subgoals involving an object or the environment, verify "
    "the corresponding state change instead of assuming success merely because "
    "the robot's motion has stopped. "
)
