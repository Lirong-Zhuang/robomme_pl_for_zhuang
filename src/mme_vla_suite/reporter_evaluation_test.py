import pytest

from mme_vla_suite.reporter_evaluation import confirm_reporter_success


def _apply_confirmation_sequence(
    predictions: list[bool | None],
    required: int,
) -> list[bool | None]:
    count = 0
    results = []
    for prediction in predictions:
        results.append(confirm_reporter_success(prediction, count, required))
        count = count + 1 if prediction is True else 0
    return results


def test_reporter_success_requires_two_consecutive_true_predictions() -> None:
    assert _apply_confirmation_sequence(
        [False, True, False, True, True],
        required=2,
    ) == [False, False, False, False, True]


def test_reporter_success_confirmation_count_one_preserves_raw_results() -> None:
    assert _apply_confirmation_sequence(
        [False, True, None],
        required=1,
    ) == [False, True, None]


def test_invalid_reporter_output_breaks_a_success_run() -> None:
    assert _apply_confirmation_sequence(
        [True, None, True, True],
        required=2,
    ) == [False, None, False, True]


def test_reporter_success_confirmation_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        confirm_reporter_success(True, 0, 0)
