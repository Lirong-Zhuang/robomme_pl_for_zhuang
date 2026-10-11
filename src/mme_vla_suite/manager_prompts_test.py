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


def test_completed_prompt_contains_model_driven_verification_steps() -> None:
    completed = _followup(True)

    assert "Classify the current subgoal" in completed
    assert '[["robot","HOLDS","<color> cube"]' in completed
    assert '[["<color> cube","IN","bin"]]' in completed
    assert '[["robot","PRESSES","button"]]' in completed
    assert "infer each RELATIONSHIP from the current image" in completed
    assert "Classification alone does not prove completion" in completed
    assert "Always output the inferred pairs" in completed
    assert "exactly one minified JSON line" in completed


def test_all_followups_retain_single_image_placeholder() -> None:
    assert _followup(True).count("<image>") == 1
    assert _followup(False).count("<image>") == 1
    assert _followup(None).count("<image>") == 1
