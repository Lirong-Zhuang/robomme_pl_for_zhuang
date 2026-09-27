from mme_vla_suite.reporter_prompts import (
    REPORTER_SYSTEM_PROMPT,
    format_reporter_user_prompt,
)
from mme_vla_suite.reporter_evaluation import _format_messages_for_history


def test_reporter_prompt_requires_visible_physical_outcome() -> None:
    user_prompt = format_reporter_user_prompt("place the object", history_size=3)

    assert "required physical end state" in REPORTER_SYSTEM_PROMPT
    assert "completed motion sequence or robot pose alone is not success" in (
        REPORTER_SYSTEM_PROMPT
    )
    assert "requested final state" in REPORTER_SYSTEM_PROMPT
    assert "absent or uncertain" in REPORTER_SYSTEM_PROMPT
    assert "required physical outcome" in user_prompt
    assert "not by motion completion" in user_prompt
    assert "absent or uncertain" in user_prompt


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
