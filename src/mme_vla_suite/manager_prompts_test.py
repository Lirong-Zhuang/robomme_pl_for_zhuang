from mme_vla_suite.prompts import get_manager_prompt


def _followup(reporter_result: bool | None) -> str:
    prompt = get_manager_prompt("qwenvl_interaction_verify_v2")
    return prompt.format_user_prompt(
        subgoal_type="simple_subgoal",
        task_goal="put a green cube into the bin",
        history_subgoals=["pick up the green cube"],
        reporter_result=reporter_result,
        has_video=False,
    )


def test_interaction_details_are_injected_only_for_reporter_true() -> None:
    completed = _followup(True)
    incomplete = _followup(False)
    missing = _followup(None)

    assert "interaction_pairs" in completed
    assert "interaction_pairs" not in incomplete
    assert "interaction_pairs" not in missing


def test_completed_prompt_contains_all_uppercase_relation_requirements() -> None:
    completed = _followup(True)

    assert '["robot","HOLDS","<target>"]' in completed
    assert '["<target>","ABOVE","table"]' in completed
    assert '["<target>","IN","<bin>"]' in completed
    assert '["robot","PRESSES","<button>"]' in completed
    assert "include them even when false or unclear" in completed
    assert "never use [] for these tasks" in completed
    assert "exactly one minified JSON line" in completed


def test_all_followups_retain_single_image_placeholder() -> None:
    assert _followup(True).count("<image>") == 1
    assert _followup(False).count("<image>") == 1
    assert _followup(None).count("<image>") == 1
