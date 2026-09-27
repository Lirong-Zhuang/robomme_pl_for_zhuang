import pytest

from mme_vla_suite.reporter_prompts import (
    PHYSICAL_OUTCOME_REPORTER_SYSTEM_PROMPT,
    REPORTER_SYSTEM_PROMPT,
    format_physical_outcome_reporter_user_prompt,
    format_reporter_eval_prompts,
    format_reporter_user_prompt,
)


def test_original_eval_variant_preserves_training_prompt() -> None:
    system_prompt, user_prompt = format_reporter_eval_prompts(
        "pick up the third red cube",
        history_size=7,
        prompt_variant="original",
    )

    assert system_prompt == REPORTER_SYSTEM_PROMPT
    assert user_prompt == format_reporter_user_prompt(
        "pick up the third red cube",
        history_size=7,
    )


def test_physical_outcome_variant_preserves_image_alignment() -> None:
    system_prompt, user_prompt = format_reporter_eval_prompts(
        "pick up the third red cube",
        history_size=7,
        prompt_variant="physical_outcome",
    )

    assert system_prompt == PHYSICAL_OUTCOME_REPORTER_SYSTEM_PROMPT
    assert user_prompt == format_physical_outcome_reporter_user_prompt(
        "pick up the third red cube",
        history_size=7,
    )
    assert user_prompt.count("<image>") == 8
    assert "Recent observation 7/7 (current observation)" in user_prompt
    assert "motion sequence" in system_prompt
    assert "requested final state" in system_prompt.lower()
    assert "uncertain" in system_prompt.lower()


def test_eval_prompt_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="Unsupported Reporter prompt variant"):
        format_reporter_eval_prompts(
            "pick up the third red cube",
            prompt_variant="unknown",
        )
