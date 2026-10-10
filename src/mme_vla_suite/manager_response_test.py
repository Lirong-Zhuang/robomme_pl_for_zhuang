import json

from mme_vla_suite.manager_response import parse_manager_response
from mme_vla_suite.manager_response import parse_verification_passed
from mme_vla_suite.manager_response import should_update_init_frame


def test_plain_manager_response_is_unchanged():
    response = "pick up the first green cube"
    assert parse_manager_response(response) == (response, None)


def test_structured_manager_response_extracts_subgoal():
    payload = {
        "interaction_pairs": [["robot", "HOLDS", "green cube"]],
        "verification_passed": True,
        "subgoal": "put it into the bin",
    }
    response = json.dumps(payload)
    assert parse_manager_response(response) == (
        "put it into the bin",
        payload,
    )


def test_fenced_structured_manager_response_is_supported():
    response = '```json\n{"interaction_pairs": [], "subgoal": "press the button"}\n```'
    subgoal, payload = parse_manager_response(response)
    assert subgoal == "press the button"
    assert payload == {
        "interaction_pairs": [],
        "subgoal": "press the button",
    }


def test_json_without_subgoal_falls_back_to_raw_response():
    response = '{"interaction_pairs": []}'
    assert parse_manager_response(response) == (response, None)


def test_verification_passed_requires_a_boolean():
    assert parse_verification_passed({"verification_passed": True}) is True
    assert parse_verification_passed({"verification_passed": False}) is False
    assert parse_verification_passed({"verification_passed": "true"}) is None
    assert parse_verification_passed(None) is None


def test_init_frame_update_requires_confirmation_when_explicitly_vetoed():
    assert should_update_init_frame(
        has_current_subgoal=True,
        subgoal_changed=True,
        update_confirmation=False,
    ) is False
    assert should_update_init_frame(
        has_current_subgoal=True,
        subgoal_changed=False,
        update_confirmation=True,
    ) is True


def test_initial_and_legacy_init_frame_behavior_is_preserved():
    assert should_update_init_frame(
        has_current_subgoal=False,
        subgoal_changed=True,
        update_confirmation=False,
    ) is True
    assert should_update_init_frame(
        has_current_subgoal=True,
        subgoal_changed=True,
        update_confirmation=None,
    ) is True
