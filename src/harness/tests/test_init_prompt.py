from __future__ import annotations

import pytest

from harness.prompts.init_prompt import build_task_init_messages
from harness.prompts.task_description import available_task_descriptions
from harness.prompts.task_description import get_task_description


def test_expected_task_descriptions_are_registered() -> None:
    assert available_task_descriptions() == (
        "BinFill",
        "MoveCube",
        "PickHighlight",
    )


@pytest.mark.parametrize(
    ("task_category", "expected_text"),
    (
        ("BinFill", "per-color placement counts"),
        ("MoveCube", "demonstrated strategy"),
        ("PickHighlight", "white highlighted area"),
    ),
)
def test_task_description_contains_defining_fact(
    task_category: str,
    expected_text: str,
) -> None:
    description = get_task_description(task_category)
    assert expected_text in description.to_prompt_text()


def test_description_lookup_is_case_insensitive() -> None:
    assert get_task_description("  binfill  ").task_category == "BinFill"


def test_unknown_description_reports_available_tasks() -> None:
    with pytest.raises(KeyError, match="BinFill, MoveCube, PickHighlight"):
        get_task_description("UnknownTask")


def test_registered_description_is_injected_into_init_prompt() -> None:
    messages = build_task_init_messages(
        task_category="PickHighlight",
        action_candidates_path="resources/action_candidates/pick_highlight.json",
        task_config_package_path="resources/task_config_packages/reference.json",
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "Task category:\nPickHighlight" in messages[1]["content"]
    assert "reference and object-memory manipulation" in messages[1]["content"]
    assert "white highlighted area" in messages[1]["content"]
