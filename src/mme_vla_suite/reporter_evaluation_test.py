import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mme_vla_suite.reporter_evaluation import aggregate_metrics
from mme_vla_suite.reporter_evaluation import evaluate_reporter_dataset
from mme_vla_suite.reporter_evaluation import evaluate_reporter_sequence
from mme_vla_suite.reporter_evaluation import parse_reporter_success
from mme_vla_suite.reporter_evaluation import score_reporter_completion


class FakeInferRequest:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeEngine:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.batch_sizes: list[int] = []
        self.requests: list[FakeInferRequest] = []

    def infer(self, requests, *, request_config):
        del request_config
        self.batch_sizes.append(len(requests))
        self.requests.extend(requests)
        return [
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=next(self.responses)))]
            )
            for _ in requests
        ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"success": true}', True),
        ('```json\n{"success": false}\n```', False),
        ('{"success": 1}', None),
        ("success", None),
    ],
)
def test_parse_reporter_success(text: str, expected: bool | None):
    assert parse_reporter_success(text) is expected


def _write_dataset(path: Path, image_paths: list[Path], labels: list[bool]) -> None:
    rows = []
    for label in labels:
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "two images"},
                    {"role": "assistant", "content": json.dumps({"success": label})},
                ],
                "images": [str(image_paths[0]), str(image_paths[1])],
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_evaluate_reporter_dataset_counts_invalid_as_incorrect(tmp_path: Path):
    images = [tmp_path / "before.png", tmp_path / "after.png"]
    for image in images:
        image.write_bytes(b"image")
    dataset_path = tmp_path / "test.jsonl"
    _write_dataset(dataset_path, images, [True, True, False, False])
    engine = FakeEngine(
        [
            '{"success": true}',
            '{"success": false}',
            '```json\n{"success": false}\n```',
            "not json",
        ]
    )
    predictions_path = tmp_path / "predictions.jsonl"

    metrics = evaluate_reporter_dataset(
        name="simple",
        dataset_path=dataset_path,
        engine=engine,
        infer_request_type=FakeInferRequest,
        request_config=object(),
        model_path="model",
        adapter_path="adapter",
        predictions_path=predictions_path,
        batch_size=3,
        progress_every=0,
    )

    assert engine.batch_sizes == [3, 1]
    assert metrics.total == 4
    assert metrics.correct == 2
    assert metrics.parsed == 3
    assert metrics.invalid == 1
    assert metrics.accuracy == 0.5
    assert metrics.valid_accuracy == pytest.approx(2 / 3)
    assert (metrics.true_positive, metrics.false_negative) == (1, 1)
    assert (metrics.true_negative, metrics.false_positive) == (1, 0)
    assert metrics.completed_recall == 0.5
    assert metrics.incomplete_recall == 0.5
    predictions = [json.loads(line) for line in predictions_path.read_text().splitlines()]
    assert [row["correct"] for row in predictions] == [True, False, True, False]
    assert predictions[-1]["predicted"] is None

    overall = aggregate_metrics([metrics])
    assert overall["correct"] == 2
    assert overall["accuracy"] == 0.5


def test_rejects_missing_assistant_label(tmp_path: Path):
    images = [tmp_path / "before.png", tmp_path / "after.png"]
    for image in images:
        image.write_bytes(b"image")
    dataset_path = tmp_path / "bad.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "missing label"}],
                "images": [str(image) for image in images],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="assistant label"):
        evaluate_reporter_dataset(
            name="simple",
            dataset_path=dataset_path,
            engine=FakeEngine([]),
            infer_request_type=FakeInferRequest,
            request_config=object(),
            model_path="model",
            adapter_path=None,
            predictions_path=tmp_path / "predictions.jsonl",
        )


def test_sequence_uses_predictions_to_update_init_and_skips_duplicates(tmp_path: Path):
    image_paths = [
        tmp_path / f"BinFill_ep0_step{step}.png"
        for step in (0, 10, 20, 30)
    ]
    for image in image_paths:
        image.write_bytes(b"image")

    rows = []
    comparisons = [
        (image_paths[0], image_paths[1], False),
        (image_paths[0], image_paths[1], False),  # Training-balance duplicate.
        (image_paths[0], image_paths[2], True),
        (image_paths[2], image_paths[3], False),
    ]
    for before, current, expected in comparisons:
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "two images"},
                    {
                        "role": "assistant",
                        "content": json.dumps({"success": expected}),
                    },
                ],
                "images": [str(before), str(current)],
            }
        )
    dataset_path = tmp_path / "simple_subgoal_train.jsonl"
    dataset_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    # The false positive at step 10 promotes step 10 to init. The false
    # negative at step 20 then leaves step 10 as init for step 30.
    engine = FakeEngine(
        [
            '{"success": true}',
            '{"success": false}',
            '{"success": false}',
        ]
    )
    predictions_path = tmp_path / "predictions.jsonl"
    metrics = evaluate_reporter_sequence(
        name="simple",
        dataset_path=dataset_path,
        engine=engine,
        infer_request_type=FakeInferRequest,
        request_config=object(),
        model_path="model",
        adapter_path="adapter",
        predictions_path=predictions_path,
        progress_every=0,
    )

    assert engine.batch_sizes == [1, 1, 1]
    assert metrics.episodes == 1
    assert metrics.total == 3
    assert metrics.duplicates_skipped == 1
    assert metrics.correct == 1
    assert [
        request.kwargs["images"][0]
        for request in engine.requests
    ] == [
        str(image_paths[0]),
        str(image_paths[1]),
        str(image_paths[1]),
    ]
    predictions = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["init_updated"] for row in predictions] == [True, False, False]
    assert [row["correct"] for row in predictions] == [False, False, True]


def _completion_record(
    call_number: int,
    *,
    expected: bool = False,
    predicted: bool | None = False,
    episode: int = 0,
) -> dict:
    return {
        "task": "BinFill",
        "episode": episode,
        "current_step": call_number * 16,
        "line_number": call_number,
        "expected": expected,
        "predicted": predicted,
    }


def test_completion_scoring_debounces_and_reports_delayed_warning():
    records = [
        _completion_record(1),
        _completion_record(2, expected=True, predicted=True),
        _completion_record(3, predicted=True),  # Same true run: ignored.
        _completion_record(4),
        _completion_record(5, expected=True),
        _completion_record(6),
        _completion_record(7),
        _completion_record(8, predicted=True),  # Three calls late: warning.
        _completion_record(9),
    ]

    report = score_reporter_completion(records)
    episode = report["episodes"][0]

    assert report["mean_completion"] == 1.0
    assert report["fully_completed_episodes"] == 1
    assert report["episodes_with_delay_warning"] == 1
    assert episode["raw_predicted_true_count"] == 3
    assert episode["debounced_predicted_true_count"] == 2
    assert episode["completed_subgoals"] == 3
    assert episode["total_subgoals"] == 3
    assert episode["transitions"][1]["status"] == "delayed_warning"
    assert episode["transitions"][1]["delay_calls"] == 3


def test_completion_scoring_stops_at_premature_trigger():
    records = [
        _completion_record(1),
        _completion_record(2, expected=True, predicted=True),
        _completion_record(3),
        _completion_record(4, predicted=True),  # Early for the next transition.
        _completion_record(5),
        _completion_record(6, expected=True),
        _completion_record(7),
    ]

    report = score_reporter_completion(records)
    episode = report["episodes"][0]

    assert episode["status"] == "premature_trigger"
    assert episode["completed_subgoals"] == 1
    assert episode["total_subgoals"] == 3
    assert episode["completion"] == pytest.approx(1 / 3)
    assert episode["failure"]["subgoal_number"] == 2


def test_completion_scoring_stalls_after_four_delayed_calls():
    records = [
        _completion_record(1, expected=True),
        _completion_record(2),
        _completion_record(3),
        _completion_record(4),
        _completion_record(5),
        _completion_record(6, predicted=True),
    ]

    report = score_reporter_completion(records)
    episode = report["episodes"][0]

    assert episode["status"] == "stalled_timeout"
    assert episode["completion"] == 0.0
    assert episode["failure"]["delay_calls"] == 5


def test_completion_scoring_penalizes_extra_true_in_final_subgoal():
    records = [
        _completion_record(1, expected=True, predicted=True),
        _completion_record(2),
        _completion_record(3, predicted=True),
    ]

    report = score_reporter_completion(records)
    episode = report["episodes"][0]

    assert episode["status"] == "final_subgoal_premature_trigger"
    assert episode["completed_subgoals"] == 1
    assert episode["total_subgoals"] == 2
    assert episode["completion"] == 0.5


def test_completion_scoring_uses_episode_macro_average_and_delay_boundaries():
    records = [
        _completion_record(1, expected=True, episode=0),
        _completion_record(2, episode=0),
        _completion_record(3, predicted=True, episode=0),  # Two calls late: full credit.
        _completion_record(1, expected=True, episode=1),
        _completion_record(2, episode=1),
        _completion_record(3, episode=1),
        _completion_record(4, episode=1),
        _completion_record(5, predicted=True, episode=1),  # Four calls late: warning.
    ]

    report = score_reporter_completion(records)

    assert report["mean_completion"] == 1.0
    assert report["fully_completed_episodes"] == 2
    assert report["episodes_with_delay_warning"] == 1
    assert report["episodes"][0]["warnings"] == []
    assert report["episodes"][1]["warnings"][0]["delay_calls"] == 4


def test_completion_scoring_allows_two_early_calls_but_not_three():
    allowed_records = [
        _completion_record(1, predicted=True),
        _completion_record(2),
        _completion_record(3, expected=True),
    ]
    rejected_records = [
        _completion_record(1, predicted=True),
        _completion_record(2),
        _completion_record(3),
        _completion_record(4, expected=True),
    ]

    allowed = score_reporter_completion(allowed_records)["episodes"][0]
    rejected = score_reporter_completion(rejected_records)["episodes"][0]

    assert allowed["status"] == "completed"
    assert allowed["completion"] == 1.0
    assert allowed["transitions"][0]["delay_calls"] == -2
    assert rejected["status"] == "premature_trigger"
    assert rejected["completion"] == 0.0
    assert rejected["failure"]["early_calls"] == 3
