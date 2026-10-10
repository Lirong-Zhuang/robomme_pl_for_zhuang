from mme_vla_suite.reporter_prompts import (
    REPORTER_SYSTEM_PROMPT,
    format_reporter_user_prompt,
)
from mme_vla_suite.reporter_evaluation import _format_messages_for_history


def test_reporter_prompt_uses_subgoal_relevant_state_evidence() -> None:
    user_prompt = format_reporter_user_prompt("place the object", history_size=3)

    assert "determine whether the current robot subgoal is complete" in (
        REPORTER_SYSTEM_PROMPT
    )
    assert "comparing observation before executing" in REPORTER_SYSTEM_PROMPT
    assert "which state evidence is relevant" in user_prompt
    assert "For robot-only subgoals" in user_prompt
    assert "robot's state or pose may be sufficient" in user_prompt
    assert "For subgoals involving an object or the environment" in user_prompt
    assert "robot's motion has stopped" in user_prompt


def test_reporter_prompt_preserves_image_order_and_count() -> None:
    user_prompt = format_reporter_user_prompt("open the object", history_size=3)

    assert user_prompt.count("<image>") == 4
    assert "Observation before executing the current subgoal: <image>" in user_prompt
    assert "Recent observation 3/3 (current observation): <image>" in user_prompt


def test_offline_evaluation_reapplies_the_shared_prompt() -> None:
    messages = [
        {"role": "system", "content": "stale training prompt"},
        {"role": "user", "content": "stale user prompt"},
    ]

    formatted = _format_messages_for_history(messages, "close the object", 2)

    assert formatted[0]["content"] == REPORTER_SYSTEM_PROMPT
    assert formatted[1]["content"] == format_reporter_user_prompt(
        "close the object",
        2,
    )
