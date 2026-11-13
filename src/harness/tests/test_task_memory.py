from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.task_memory import SubgoalResolver
from harness.task_memory import TaskMemory


ALLOWED_ACTIONS = {"solve_pickup", "solve_put_into_bin", "solve_button"}


def _task_config() -> dict:
    return {
        "schema_version": "task_config.v1",
        "subgoals": [
            {
                "subgoal_key": "pickup_cube",
                "template": "pick up the {ordinal} {color} cube",
                "action_name": "solve_pickup",
                "acceptance": {
                    "completed": "The intended {color} cube is held and lifted.",
                    "incomplete": "The intended {color} cube is not yet lifted.",
                    "failed": "A wrong cube is grasped.",
                },
            },
            {
                "subgoal_key": "put_into_bin",
                "template": "put it into the bin",
                "action_name": "solve_put_into_bin",
                "acceptance": {
                    "completed": "The cube is released into the bin.",
                    "incomplete": "The cube remains held.",
                    "failed": "The cube is released outside the bin.",
                },
            },
            {
                "subgoal_key": "press_button",
                "template": "press the button",
                "action_name": "solve_button",
                "acceptance": {
                    "completed": "The button completes a press cycle.",
                    "incomplete": "The press is still ongoing.",
                    "failed": "The press misses the button.",
                },
            },
        ],
    }


def _pickup_payload() -> dict:
    return {
        "subgoal_key": "pickup_cube",
        "subgoal": "pick up the first red cube",
        "arguments": {"ordinal": "first", "color": "red"},
    }


def _put_payload() -> dict:
    return {
        "subgoal_key": "put_into_bin",
        "subgoal": "put it into the bin",
        "arguments": {},
    }


def _first_plan_output() -> dict:
    return {
        "subgoal_sequence": [_pickup_payload(), _put_payload()],
        "next_subgoal_key": "pickup_cube",
        "next_subgoal": "pick up the first red cube",
        "arguments": {"ordinal": "first", "color": "red"},
    }


def _memory(tmp_path: Path) -> TaskMemory:
    return TaskMemory(tmp_path / "episode", _task_config(), ALLOWED_ACTIONS)


def _keyframe_memory(tmp_path: Path) -> TaskMemory:
    config = _task_config()
    config["keyframes"] = [
        {
            "views": ["overall"],
            "save_when": "The transient highlight is visible.",
            "capture_mode": "single",
            "frame_count": 1,
            "interval": 16,
            "interval_unit": "frame",
            "save_description": True,
            "description_instruction": "Describe the highlighted cubes.",
            "task_state_field": "persistent_facts.keyframes.highlight_reveal",
        }
    ]
    return TaskMemory(tmp_path / "episode", config, ALLOWED_ACTIONS)


def _sequence_keyframe_memory(tmp_path: Path) -> TaskMemory:
    config = _task_config()
    config["keyframes"] = [
        {
            "views": ["overall"],
            "save_when": "The tracked containers are moving or settling.",
            "stop_when": "All tracked containers have settled.",
            "capture_mode": "sequence",
            "frame_count": 2,
            "interval": 16,
            "interval_unit": "frame",
            "save_description": True,
            "description_instruction": "Describe the tracked container positions.",
            "task_state_field": "persistent_facts.keyframes.container_tracking",
        }
    ]
    return TaskMemory(tmp_path / "episode", config, ALLOWED_ACTIONS)


def test_resolver_maps_key_to_action_and_instantiated_acceptance() -> None:
    resolver = SubgoalResolver(_task_config(), ALLOWED_ACTIONS)

    resolved = resolver.resolve_planner_payload(_pickup_payload())

    assert resolved.action_name == "solve_pickup"
    assert resolved.acceptance["completed"] == (
        "The intended red cube is held and lifted."
    )
    assert resolved.planner_payload() == _pickup_payload()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "subgoal_key": "unknown",
                "subgoal": "pick up the first red cube",
                "arguments": {},
            },
            "unknown subgoal_key",
        ),
        (
            {
                "subgoal_key": "pickup_cube",
                "subgoal": "pick up the blue cube",
                "arguments": {"ordinal": "first", "color": "red"},
            },
            "does not match",
        ),
        (
            {
                "subgoal_key": "pickup_cube",
                "subgoal": "pick up the first red cube",
                "arguments": {"color": "red"},
            },
            "missing argument",
        ),
    ],
)
def test_resolver_rejects_invalid_planner_payload(
    payload: dict,
    message: str,
) -> None:
    resolver = SubgoalResolver(_task_config(), ALLOWED_ACTIONS)

    with pytest.raises(ValueError, match=message):
        resolver.resolve_planner_payload(payload)


def test_resolver_rejects_vlm_supplied_action_name() -> None:
    resolver = SubgoalResolver(_task_config(), ALLOWED_ACTIONS)
    payload = {**_pickup_payload(), "action_name": "solve_pickup"}

    with pytest.raises(ValueError, match="must contain exactly"):
        resolver.resolve_planner_payload(payload)


def test_resolver_rejects_duplicate_key_and_untrusted_action() -> None:
    duplicate = _task_config()
    duplicate["subgoals"][1]["subgoal_key"] = "pickup_cube"
    with pytest.raises(ValueError, match="duplicate subgoal_key"):
        SubgoalResolver(duplicate, ALLOWED_ACTIONS)

    untrusted = _task_config()
    untrusted["subgoals"][0]["action_name"] = "arbitrary_python"
    with pytest.raises(ValueError, match="untrusted action"):
        SubgoalResolver(untrusted, ALLOWED_ACTIONS)


def test_first_plan_creates_state_and_append_only_event(tmp_path: Path) -> None:
    memory = _memory(tmp_path)

    state = memory.commit_first_plan(
        task_goal="Put one red cube into the bin.",
        planner_output=_first_plan_output(),
    )

    assert state["current_subgoal"]["subgoal_key"] == "pickup_cube"
    assert state["current_subgoal"]["instance_id"] == "sg-0001"
    assert state["plan"]["sequence"][1]["subgoal_key"] == "put_into_bin"
    assert state["event_cursor"] == 1

    execution = memory.current_execution()
    assert execution is not None
    assert execution["action_name"] == "solve_pickup"
    assert execution["acceptance"]["completed"] == (
        "The intended red cube is held and lifted."
    )

    events = memory.query_events()
    assert [event["type"] for event in events] == ["plan_created"]
    assert json.loads((memory.episode_dir / "task_state.json").read_text()) == state


def test_current_key_resolves_trusted_action_callable(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    memory.commit_first_plan(
        task_goal="Put one red cube into the bin.",
        planner_output=_first_plan_output(),
    )

    def pickup() -> str:
        return "picked"

    action = memory.resolve_current_action({"solve_pickup": pickup})

    assert action is pickup
    assert action() == "picked"


def test_action_events_do_not_mark_subgoal_completed(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    memory.commit_first_plan(
        task_goal="Put one red cube into the bin.",
        planner_output=_first_plan_output(),
    )

    started = memory.record_action_started(request_id="request-1")
    finished = memory.record_action_finished(
        runtime_status="returned",
        artifact_refs=["observations/0001.json"],
    )

    state = memory.load_state()
    assert state["progress"]["completed_subgoal_ids"] == []
    assert state["current_subgoal"]["subgoal_key"] == "pickup_cube"
    assert started["action_name"] == "solve_pickup"
    assert finished["runtime_status"] == "returned"


def test_followup_inputs_include_compact_state_and_current_acceptance(
    tmp_path: Path,
) -> None:
    memory = _memory(tmp_path)
    memory.commit_first_plan(
        task_goal="Put one red cube into the bin.",
        planner_output=_first_plan_output(),
    )

    inputs = memory.build_followup_inputs(
        verification_parameters={"k": 3},
        subgoal_start_observations=["start"],
        recent_observations=["recent"],
    )

    assert inputs["current_subgoal_key"] == "pickup_cube"
    assert inputs["acceptance_criteria"]["completed"] == (
        "The intended red cube is held and lifted."
    )
    assert inputs["task_state"]["event_cursor"] == 1
    assert "robot_state" not in inputs
    assert "action_name" not in inputs
    assert "keyframe_policy" not in inputs


def test_keyframe_task_selects_trusted_observation_id_and_saves_path(
    tmp_path: Path,
) -> None:
    memory = _keyframe_memory(tmp_path)
    first_inputs = memory.build_first_planning_inputs(
        task_goal="Pick the highlighted cube.",
        current_observations=["initial_overall"],
    )
    assert first_inputs["keyframe_policy"]["task_config_ref"] == (
        "task_config.keyframes[0]"
    )
    assert first_inputs["keyframe_policy"]["save_when"] == (
        "The transient highlight is visible."
    )
    assert first_inputs["saved_keyframe"] is None
    first_output = {**_first_plan_output(), "keyframe_save": None}
    memory.commit_first_plan(
        task_goal="Pick the highlighted cube.",
        planner_output=first_output,
    )

    with pytest.raises(ValueError, match="current_observation"):
        memory.build_followup_inputs(
            verification_parameters={"k": 3},
            subgoal_start_observations=["start"],
            recent_observations=["sliding_1_overall"],
        )

    inputs = memory.build_followup_inputs(
        verification_parameters={"k": 3},
        subgoal_start_observations=["start"],
        recent_observations=["sliding_1_overall"],
        current_observation="current_overall",
    )
    assert inputs["current_observation"] == "current_overall"
    assert inputs["keyframe_policy"]["task_config_ref"] == (
        "task_config.keyframes[0]"
    )
    assert inputs["keyframe_policy"]["save_when"] == (
        "The transient highlight is visible."
    )
    assert inputs["saved_keyframe"] is None

    memory.commit_followup(
        {
            "status": "completed",
            "description": "The highlight is visible on the red cube at left.",
            "next_subgoal_key": "put_into_bin",
            "next_subgoal": "put it into the bin",
            "arguments": {},
            "keyframe_save": "current_overall",
        }
    )
    state = memory.save_keyframe_selection(
        keyframe_save="current_overall",
        observation_paths={
            "sliding_1_overall": "/episode/frames/0001.png",
            "current_overall": "/episode/frames/0002.png",
        },
        description="The highlight is visible on the red cube at left.",
    )

    saved = state["persistent_facts"]["keyframes"]["highlight_reveal"]
    assert saved == {
        "observation_id": "current_overall",
        "image_path": "/episode/frames/0002.png",
        "description": "The highlight is visible on the red cube at left.",
        "policy_index": 0,
    }
    events = memory.query_events(event_type="task_state_field_updated")
    assert events[-1]["source"] == "keyframe_observation:current_overall"

    later_inputs = memory.build_followup_inputs(
        verification_parameters={"k": 3},
        subgoal_start_observations=["start"],
        recent_observations=["sliding_2_overall"],
        current_observation="current_overall_later",
    )
    assert later_inputs["saved_keyframe"] == {
        "task_config_ref": "task_config.keyframes[0]",
        "task_state_ref": "persistent_facts.keyframes.highlight_reveal",
        "observation_id": "current_overall",
        "image": "<image>",
        "image_path": "/episode/frames/0002.png",
        "description": "The highlight is visible on the red cube at left.",
    }


def test_video_subgoal_list_is_an_explicit_followup_input(tmp_path: Path) -> None:
    config = _task_config()
    config["planning"] = {
        "information_sources": ["task_goal", "video_subgoal_list"]
    }
    memory = TaskMemory(tmp_path / "episode", config, ALLOWED_ACTIONS)
    memory.commit_first_plan(
        task_goal="Move the cube as demonstrated.",
        planner_output=_first_plan_output(),
    )

    inputs = memory.build_followup_inputs(
        verification_parameters={"k": 3},
        subgoal_start_observations=["start"],
        recent_observations=["recent"],
    )

    assert inputs["video_subgoal_list"] == [
        {
            "subgoal_key": "pickup_cube",
            "subgoal": "pick up the first red cube",
            "arguments": {"ordinal": "first", "color": "red"},
        },
        {
            "subgoal_key": "put_into_bin",
            "subgoal": "put it into the bin",
            "arguments": {},
        },
    ]


def test_keyframe_selection_rejects_missing_or_untrusted_ids(tmp_path: Path) -> None:
    memory = _keyframe_memory(tmp_path)
    with pytest.raises(ValueError, match="keyframe_save"):
        memory.commit_first_plan(
            task_goal="Pick the highlighted cube.",
            planner_output=_first_plan_output(),
        )

    memory.commit_first_plan(
        task_goal="Pick the highlighted cube.",
        planner_output={**_first_plan_output(), "keyframe_save": None},
    )
    with pytest.raises(ValueError, match="not a supplied observation ID"):
        memory.save_keyframe_selection(
            keyframe_save="invented_frame",
            observation_paths={"current_overall": "/episode/frames/0002.png"},
        )


def test_sequence_keyframe_appends_until_frame_count(tmp_path: Path) -> None:
    memory = _sequence_keyframe_memory(tmp_path)
    memory.commit_first_plan(
        task_goal="Track the containers.",
        planner_output={**_first_plan_output(), "keyframe_save": None},
    )

    state = memory.save_keyframe_selection(
        keyframe_save="overall_1",
        observation_paths={"overall_1": "/episode/frames/0001.png"},
        description="The tracked container is moving from left toward center.",
    )
    assert len(state["persistent_facts"]["keyframes"]["container_tracking"]) == 1

    state = memory.save_keyframe_selection(
        keyframe_save="overall_2",
        observation_paths={"overall_2": "/episode/frames/0002.png"},
        description="The tracked container has settled at center-right.",
    )
    saved = state["persistent_facts"]["keyframes"]["container_tracking"]
    assert [item["observation_id"] for item in saved] == ["overall_1", "overall_2"]

    inputs = memory.build_followup_inputs(
        verification_parameters={"k": 3},
        subgoal_start_observations=["start"],
        recent_observations=["recent"],
        current_observation="current_overall",
    )
    assert inputs["saved_keyframe"] == {
        "task_config_ref": "task_config.keyframes[0]",
        "task_state_ref": "persistent_facts.keyframes.container_tracking",
        "capture_mode": "sequence",
        "saved_count": 2,
        "frame_count_upper_bound": 2,
        "capacity_reached": True,
        "stop_when": "All tracked containers have settled.",
        "observations": [
            {
                "observation_id": "overall_1",
                "image": "<image>",
                "image_path": "/episode/frames/0001.png",
                "description": (
                    "The tracked container is moving from left toward center."
                ),
            },
            {
                "observation_id": "overall_2",
                "image": "<image>",
                "image_path": "/episode/frames/0002.png",
                "description": "The tracked container has settled at center-right.",
            },
        ],
    }

    with pytest.raises(RuntimeError, match="frame_count upper bound"):
        memory.save_keyframe_selection(
            keyframe_save="overall_3",
            observation_paths={"overall_3": "/episode/frames/0003.png"},
            description="Another later observation.",
        )


def test_keyframe_description_is_required_by_policy(tmp_path: Path) -> None:
    memory = _sequence_keyframe_memory(tmp_path)
    memory.commit_first_plan(
        task_goal="Track the containers.",
        planner_output={**_first_plan_output(), "keyframe_save": None},
    )

    with pytest.raises(ValueError, match="description is required"):
        memory.save_keyframe_selection(
            keyframe_save="overall_1",
            observation_paths={"overall_1": "/episode/frames/0001.png"},
        )


def test_incomplete_retains_current_subgoal_and_increments_attempt(
    tmp_path: Path,
) -> None:
    memory = _memory(tmp_path)
    memory.commit_first_plan(
        task_goal="Put one red cube into the bin.",
        planner_output=_first_plan_output(),
    )

    state = memory.commit_followup(
        {
            "status": "incomplete",
            "description": "The cube remains on the table.",
            "next_subgoal_key": "pickup_cube",
            "next_subgoal": "pick up the first red cube",
            "arguments": {"ordinal": "first", "color": "red"},
        }
    )

    assert state["current_subgoal"]["instance_id"] == "sg-0001"
    assert state["current_subgoal"]["attempt"] == 2
    assert state["progress"]["completed_subgoal_ids"] == []


def test_completed_advances_and_resolves_next_action(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    memory.commit_first_plan(
        task_goal="Put one red cube into the bin.",
        planner_output=_first_plan_output(),
    )

    state = memory.commit_followup(
        {
            "status": "completed",
            "description": "The red cube is visibly held above the table.",
            "next_subgoal_key": "put_into_bin",
            "next_subgoal": "put it into the bin",
            "arguments": {},
        }
    )

    assert state["progress"]["completed_subgoal_ids"] == ["sg-0001"]
    assert state["current_subgoal"]["instance_id"] == "sg-0002"
    execution = memory.current_execution()
    assert execution is not None
    assert execution["action_name"] == "solve_put_into_bin"


def test_completed_stop_has_no_current_action(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    memory.commit_first_plan(
        task_goal="Put one red cube into the bin.",
        planner_output=_first_plan_output(),
    )
    memory.commit_followup(
        {
            "status": "completed",
            "description": "The red cube is held.",
            "next_subgoal_key": "put_into_bin",
            "next_subgoal": "put it into the bin",
            "arguments": {},
        }
    )

    state = memory.commit_followup(
        {
            "status": "completed",
            "description": "The cube is released into the bin.",
            "next_subgoal_key": "STOP",
            "next_subgoal": "STOP",
            "arguments": {},
        }
    )

    assert state["current_subgoal"] is None
    assert state["progress"]["completed_subgoal_ids"] == ["sg-0001", "sg-0002"]
    assert memory.current_execution() is None


def test_failed_retry_does_not_mark_completion(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    memory.commit_first_plan(
        task_goal="Put one red cube into the bin.",
        planner_output=_first_plan_output(),
    )

    state = memory.commit_followup(
        {
            "status": "failed",
            "description": "The cube was dropped but remains reachable.",
            "next_subgoal_key": "pickup_cube",
            "next_subgoal": "pick up the first red cube",
            "arguments": {"ordinal": "first", "color": "red"},
        }
    )

    assert state["progress"]["completed_subgoal_ids"] == []
    assert state["current_subgoal"]["instance_id"] == "sg-0001"
    assert state["current_subgoal"]["attempt"] == 2


def test_dynamic_followup_subgoal_is_appended_to_prefix_plan(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    first = {
        "subgoal_sequence": [
            {
                "subgoal_key": "press_button",
                "subgoal": "press the button",
                "arguments": {},
            }
        ],
        "next_subgoal_key": "press_button",
        "next_subgoal": "press the button",
        "arguments": {},
    }
    memory.commit_first_plan(task_goal="Reveal the target.", planner_output=first)

    state = memory.commit_followup(
        {
            "status": "completed",
            "description": "The button completed a press cycle.",
            "next_subgoal_key": "pickup_cube",
            "next_subgoal": "pick up the first red cube",
            "arguments": {"ordinal": "first", "color": "red"},
        }
    )

    assert len(state["plan"]["sequence"]) == 2
    assert state["current_subgoal"]["instance_id"] == "sg-0002"
    assert state["current_subgoal"]["subgoal_key"] == "pickup_cube"


def test_event_query_filters_by_type_and_subgoal(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    memory.commit_first_plan(
        task_goal="Put one red cube into the bin.",
        planner_output=_first_plan_output(),
    )
    memory.record_action_started()
    memory.record_action_finished(runtime_status="returned")

    events = memory.query_events(
        event_type="action_finished",
        subgoal_id="sg-0001",
    )

    assert len(events) == 1
    assert events[0]["action_name"] == "solve_pickup"


def test_configured_task_state_field_updates_state_and_event_log(
    tmp_path: Path,
) -> None:
    memory = _memory(tmp_path)
    memory.commit_first_plan(
        task_goal="Put one red cube into the bin.",
        planner_output=_first_plan_output(),
    )

    state = memory.set_task_state_field(
        "persistent_facts.keyframes.highlight_reveal",
        {"targets": ["red cube at back left"]},
        source="keyframe-0001",
    )

    assert state["persistent_facts"]["keyframes"]["highlight_reveal"] == {
        "targets": ["red cube at back left"]
    }
    events = memory.query_events(event_type="task_state_field_updated")
    assert events[0]["field_path"] == (
        "persistent_facts.keyframes.highlight_reveal"
    )


def test_task_state_is_rejected_with_different_config(tmp_path: Path) -> None:
    episode_dir = tmp_path / "episode"
    memory = TaskMemory(episode_dir, _task_config(), ALLOWED_ACTIONS)
    memory.commit_first_plan(
        task_goal="Put one red cube into the bin.",
        planner_output=_first_plan_output(),
    )

    changed = _task_config()
    changed["subgoals"][0]["acceptance"]["completed"] = "Different criterion."
    reopened = TaskMemory(episode_dir, changed, ALLOWED_ACTIONS)

    with pytest.raises(RuntimeError, match="different task config"):
        reopened.current_execution()

