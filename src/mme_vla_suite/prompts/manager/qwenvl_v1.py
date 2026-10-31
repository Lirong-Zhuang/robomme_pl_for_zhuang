"""QwenVL Manager prompt shared by Trinity v2.1 and v2.2."""

VERSION = "qwenvl_v1"

SIMPLE_SYSTEM_PROMPT = (
    "You are a helpful assistant to help guide the robot to complete the task "
    "by predicting a sequence of language subgoals"
)

GROUNDED_SYSTEM_PROMPT = (
    "You are a helpful assistant to help guide the robot to complete the task "
    "by predicting a sequence of grounded language subgoals"
)

INITIAL_USER_PROMPT_TEMPLATE = (
    "{video_prefix}The task goal is: {task_goal}\n"
    "This is the initial turn for prediction\n"
    "<image>What's the next {subgoal_name} based on current observation?"
)

FOLLOWUP_USER_PROMPT_TEMPLATE = (
    "{video_prefix}The task goal is: {task_goal}\n"
    "The history of previous predicted {subgoal_name_plural} are: {history}\n"
    "{reporter_text}\n"
    "<image>What's the next {subgoal_name} based on current observation and the result from the Reporter? "
    "If the Reporter determines that the last subgoal is not complete, output the same subgoal."
)

REPORTER_COMPLETED_TEMPLATE = (
    "The Reporter determined that the last predicted {subgoal_name} has been completed."
)
REPORTER_INCOMPLETE_TEMPLATE = (
    "The Reporter determined that the last predicted {subgoal_name} has not been completed."
)
REPORTER_MISSING_TEMPLATE = (
    "The Reporter did not provide a result for the last predicted {subgoal_name}."
)
