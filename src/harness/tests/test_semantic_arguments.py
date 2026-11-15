from __future__ import annotations

import pytest

from harness.actions.semantic_arguments import resolve_action_arguments


def test_insert_side_comes_from_contained_word_not_whole_subgoal() -> None:
    assert resolve_action_arguments(
        "insert_peg",
        subgoal="carefully insert the selected peg from the LEFT side of the box",
        arguments={"planning_note": "retained"},
    ) == {"planning_note": "retained", "direction": -1}

    assert resolve_action_arguments(
        "insert_peg",
        subgoal="insert from the right side now",
    ) == {"direction": 1}


def test_lateral_side_rejects_missing_or_conflicting_words() -> None:
    with pytest.raises(ValueError, match="exactly one lateral side"):
        resolve_action_arguments("insert_peg", subgoal="insert the peg")

    with pytest.raises(ValueError, match="exactly one lateral side"):
        resolve_action_arguments(
            "insert_peg",
            subgoal="move from left to right before insertion",
        )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("move forward", "forward"),
        ("move to the target on the right", "right"),
        ("move backward and left", "backward-left"),
        ("continue forward-right", "forward-right"),
    ],
)
def test_pattern_direction_is_composed_from_independent_tokens(
    text: str,
    expected: str,
) -> None:
    assert resolve_action_arguments(
        "solve_pattern_direction",
        subgoal=text,
    )["direction"] == expected


@pytest.mark.parametrize(
    ("text", "side", "direction"),
    [
        (
            "move to the nearest left target by circling clockwise",
            "left",
            "clockwise",
        ),
        (
            "circle counterclockwise around the stick toward the right target",
            "right",
            "counterclockwise",
        ),
    ],
)
def test_route_arguments_are_extracted_by_word_tokens(
    text: str,
    side: str,
    direction: str,
) -> None:
    resolved = resolve_action_arguments(
        "solve_swingonto_with_direction",
        subgoal=text,
    )
    assert resolved["side"] == side
    assert resolved["direction"] == direction
