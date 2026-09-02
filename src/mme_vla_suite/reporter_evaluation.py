"""Offline evaluation utilities for the RoboMME Reporter.

The evaluator consumes the same JSONL format produced by
``build_reporter_dataset.py``.  The final assistant message is treated as the
ground-truth label and is removed before the request is sent to the model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any
from typing import Protocol


def resolve_reporter_test_dataset(testset_path: Path, subgoal_type: str) -> Path:
    """Resolve a Reporter test JSONL from a file, split root, or dataset root."""
    path = testset_path.expanduser().resolve()
    if path.is_file():
        return path
    search_roots = (
        path,
        path / "reporter_qwenvl",
        path / "testset",
        path / "testset" / "reporter_qwenvl",
    )
    test_filename = f"{subgoal_type}_subgoal_test.jsonl"
    candidates = [root / test_filename for root in search_roots]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find a {subgoal_type} Reporter JSONL under {path}. Tried:\n  {tried}"
    )


def parse_reporter_success(response: str) -> bool | None:
    """Parse the strict JSON response used by both live and offline Reporter."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        success = json.loads(text).get("success")
    except (json.JSONDecodeError, AttributeError):
        return None
    return success if isinstance(success, bool) else None


@dataclass(frozen=True)
class ReporterSample:
    """One labelled Reporter request."""

    line_number: int
    messages: list[dict[str, Any]]
    images: list[str]
    expected: bool

    def as_request(self) -> dict[str, Any]:
        return {"messages": self.messages, "images": self.images}


@dataclass
class ReporterMetrics:
    """Classification and output-format metrics for one dataset."""

    name: str
    dataset_path: str
    model_path: str
    adapter_path: str | None
    total: int = 0
    correct: int = 0
    parsed: int = 0
    invalid: int = 0
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0
    completed_total: int = 0
    incomplete_total: int = 0
    episodes: int = 0
    duplicates_skipped: int = 0

    @property
    def accuracy(self) -> float | None:
        """Accuracy over every row; invalid outputs count as incorrect."""
        return self.correct / self.total if self.total else None

    @property
    def valid_accuracy(self) -> float | None:
        """Accuracy only among responses that contain a valid boolean label."""
        return self.correct / self.parsed if self.parsed else None

    @property
    def parse_rate(self) -> float | None:
        return self.parsed / self.total if self.total else None

    @property
    def completed_recall(self) -> float | None:
        return self.true_positive / self.completed_total if self.completed_total else None

    @property
    def incomplete_recall(self) -> float | None:
        return self.true_negative / self.incomplete_total if self.incomplete_total else None

    def update(self, *, expected: bool, predicted: bool | None) -> bool:
        self.total += 1
        if expected:
            self.completed_total += 1
        else:
            self.incomplete_total += 1
        if predicted is None:
            self.invalid += 1
            return False

        self.parsed += 1
        correct = predicted == expected
        self.correct += int(correct)
        if expected and predicted:
            self.true_positive += 1
        elif not expected and not predicted:
            self.true_negative += 1
        elif not expected and predicted:
            self.false_positive += 1
        else:
            self.false_negative += 1
        return correct

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "accuracy": self.accuracy,
                "valid_accuracy": self.valid_accuracy,
                "parse_rate": self.parse_rate,
                "completed_recall": self.completed_recall,
                "incomplete_recall": self.incomplete_recall,
            }
        )
        return result


class ReporterEngine(Protocol):
    """Minimal subset of the ms-swift engine used by the evaluator."""

    def infer(self, requests: list[Any], *, request_config: Any) -> Sequence[Any]: ...


@dataclass(frozen=True)
class ReporterFrame:
    """One labelled frame comparison positioned within an episode."""

    sample: ReporterSample
    task_name: str
    episode_id: int
    before_step: int
    current_step: int


def score_reporter_completion(
    records: Sequence[dict[str, Any]],
    *,
    early_tolerance_frames: int = 2,
    full_credit_delay_calls: int = 2,
    maximum_delay_calls: int = 4,
) -> dict[str, Any]:
    """Score causal, prefix-safe subgoal completion for every episode.

    A ground-truth ``true`` marks a transition to the next subgoal. Predicted
    ``true`` runs are debounced to their rising edge. Predictions are consumed
    in order: an event earlier than ``early_tolerance_frames``, a missing
    event, or an event later than ``maximum_delay_calls`` stops progress for
    that episode. Delays beyond ``full_credit_delay_calls`` remain completed
    but are reported as warnings.

    An episode with N expected transitions contains N + 1 subgoals. The final
    subgoal receives credit only when all transitions were safe and no extra
    rising-edge ``true`` occurs afterward.
    """
    if early_tolerance_frames < 0:
        raise ValueError("early_tolerance_frames must be non-negative")
    if full_credit_delay_calls < 0:
        raise ValueError("full_credit_delay_calls must be non-negative")
    if maximum_delay_calls < full_credit_delay_calls:
        raise ValueError(
            "maximum_delay_calls must be at least full_credit_delay_calls"
        )

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        try:
            episode_key = (str(record["task"]), int(record["episode"]))
            int(record["current_step"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Completion scoring records require task, episode, and current_step"
            ) from error
        grouped.setdefault(episode_key, []).append(record)

    episodes: list[dict[str, Any]] = []
    for (task_name, episode_id), episode_records in sorted(grouped.items()):
        ordered = sorted(
            episode_records,
            key=lambda record: (int(record["current_step"]), int(record["line_number"])),
        )

        expected_events: list[dict[str, int]] = []
        predicted_events: list[dict[str, int]] = []
        raw_predicted_true_count = 0
        previous_was_true = False
        for call_index, record in enumerate(ordered):
            event = {
                "call_number": call_index + 1,
                "current_step": int(record["current_step"]),
                "line_number": int(record["line_number"]),
            }
            if record.get("expected") is True:
                expected_events.append(event)

            predicted_true = record.get("predicted") is True
            if predicted_true:
                raw_predicted_true_count += 1
                if not previous_was_true:
                    predicted_events.append(event)
            previous_was_true = predicted_true

        transitions: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        predicted_index = 0
        completed_subgoals = 0
        status = "completed"
        failure: dict[str, Any] | None = None

        for transition_index, expected_event in enumerate(expected_events, start=1):
            if predicted_index >= len(predicted_events):
                status = "stalled_missing_true"
                failure = {
                    "type": status,
                    "subgoal_number": transition_index,
                    "expected_event": expected_event,
                }
                transitions.append(
                    {
                        "subgoal_number": transition_index,
                        "status": status,
                        "expected_event": expected_event,
                        "predicted_event": None,
                        "delay_calls": None,
                        "delay_frames": None,
                    }
                )
                break

            predicted_event = predicted_events[predicted_index]
            delay_calls = (
                predicted_event["call_number"] - expected_event["call_number"]
            )
            delay_frames = (
                predicted_event["current_step"] - expected_event["current_step"]
            )
            detail = {
                "subgoal_number": transition_index,
                "expected_event": expected_event,
                "predicted_event": predicted_event,
                "delay_calls": delay_calls,
                "delay_frames": delay_frames,
            }

            if delay_frames < -early_tolerance_frames:
                status = "premature_trigger"
                detail["status"] = status
                transitions.append(detail)
                failure = {
                    "type": status,
                    "subgoal_number": transition_index,
                    "expected_event": expected_event,
                    "predicted_event": predicted_event,
                    "early_calls": -delay_calls,
                    "early_frames": -delay_frames,
                }
                break
            if delay_calls > maximum_delay_calls:
                status = "stalled_timeout"
                detail["status"] = status
                transitions.append(detail)
                failure = {
                    "type": status,
                    "subgoal_number": transition_index,
                    "expected_event": expected_event,
                    "predicted_event": predicted_event,
                    "delay_calls": delay_calls,
                    "delay_frames": delay_frames,
                }
                break

            predicted_index += 1
            completed_subgoals += 1
            if delay_calls > full_credit_delay_calls:
                detail["status"] = "delayed_warning"
                warning = {
                    "type": "delayed_warning",
                    "subgoal_number": transition_index,
                    "delay_calls": delay_calls,
                    "delay_frames": delay_frames,
                }
                warnings.append(warning)
            else:
                detail["status"] = "completed"
            transitions.append(detail)

        if failure is None:
            if predicted_index < len(predicted_events):
                status = "final_subgoal_premature_trigger"
                predicted_event = predicted_events[predicted_index]
                failure = {
                    "type": status,
                    "subgoal_number": len(expected_events) + 1,
                    "predicted_event": predicted_event,
                }
            else:
                # The dataset terminates without another Reporter request, so a
                # clean final segment is treated as the final completed subgoal.
                completed_subgoals += 1

        total_subgoals = len(expected_events) + 1
        completion = completed_subgoals / total_subgoals
        episodes.append(
            {
                "task": task_name,
                "episode": episode_id,
                "completion": completion,
                "completed_subgoals": completed_subgoals,
                "total_subgoals": total_subgoals,
                "status": status,
                "total_reporter_calls": len(ordered),
                "expected_true_count": len(expected_events),
                "raw_predicted_true_count": raw_predicted_true_count,
                "debounced_predicted_true_count": len(predicted_events),
                "expected_events": expected_events,
                "predicted_events": predicted_events,
                "transitions": transitions,
                "warnings": warnings,
                "failure": failure,
            }
        )

    mean_completion = (
        sum(episode["completion"] for episode in episodes) / len(episodes)
        if episodes
        else None
    )
    return {
        "scoring": {
            "early_tolerance_frames": early_tolerance_frames,
            "full_credit_delay_calls": full_credit_delay_calls,
            "maximum_delay_calls": maximum_delay_calls,
            "early_trigger_policy": "fatal_beyond_tolerance",
            "consecutive_true_policy": "keep_first_rising_edge",
        },
        "episode_count": len(episodes),
        "mean_completion": mean_completion,
        "fully_completed_episodes": sum(
            episode["completion"] == 1.0 for episode in episodes
        ),
        "premature_trigger_episodes": sum(
            episode["status"]
            in {"premature_trigger", "final_subgoal_premature_trigger"}
            for episode in episodes
        ),
        "stalled_episodes": sum(
            episode["status"] in {"stalled_missing_true", "stalled_timeout"}
            for episode in episodes
        ),
        "episodes_with_delay_warning": sum(
            bool(episode["warnings"]) for episode in episodes
        ),
        "episodes": episodes,
    }


_REPORTER_FRAME_NAME = re.compile(
    r"^(?P<task>.+)_ep(?P<episode>\d+)_step(?P<step>\d+)\.[^.]+$"
)


def _resolve_image_path(raw_path: str, dataset_path: Path, image_root: Path | None) -> Path:
    path = Path(raw_path).expanduser()
    candidates: list[Path] = [path]
    if image_root is not None:
        candidates.extend([image_root / path.name, image_root / raw_path])
    candidates.extend(
        [
            dataset_path.parent / raw_path,
            dataset_path.parent / "images" / path.name,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not resolve image {raw_path!r} referenced by {dataset_path}. Tried: {tried}"
    )


def load_reporter_samples(
    dataset_path: str | Path,
    *,
    image_root: str | Path | None = None,
    max_samples: int | None = None,
) -> list[ReporterSample]:
    """Load and validate Reporter JSONL rows."""
    path = Path(dataset_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Reporter test dataset not found: {path}")
    resolved_image_root = Path(image_root).expanduser().resolve() if image_root else None
    samples: list[ReporterSample] = []

    with path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if max_samples is not None and len(samples) >= max_samples:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error

            messages = row.get("messages")
            images = row.get("images")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"Missing messages list at {path}:{line_number}")
            if not isinstance(images, list) or len(images) != 2:
                raise ValueError(f"Expected exactly two images at {path}:{line_number}")

            label_message = messages[-1]
            if not isinstance(label_message, dict) or label_message.get("role") != "assistant":
                raise ValueError(f"Last message must be the assistant label at {path}:{line_number}")
            label_content = label_message.get("content", "")
            if not isinstance(label_content, str):
                raise ValueError(f"Reporter label is not text at {path}:{line_number}")
            expected = parse_reporter_success(label_content)
            if expected is None:
                raise ValueError(f"Invalid Reporter label at {path}:{line_number}: {label_message!r}")

            request_messages = messages[:-1]
            if not request_messages:
                raise ValueError(f"No request messages remain at {path}:{line_number}")
            resolved_images = [
                str(_resolve_image_path(str(image), path, resolved_image_root)) for image in images
            ]
            samples.append(
                ReporterSample(
                    line_number=line_number,
                    messages=request_messages,
                    images=resolved_images,
                    expected=expected,
                )
            )

    if not samples:
        raise ValueError(f"Reporter test dataset contains no samples: {path}")
    return samples


def _response_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as error:
        raise TypeError(f"Unexpected Reporter inference response: {response!r}") from error
    if not isinstance(content, str):
        raise TypeError(f"Reporter response content is not text: {content!r}")
    return content


def _parse_reporter_frame_path(path: str, *, line_number: int) -> tuple[str, int, int]:
    match = _REPORTER_FRAME_NAME.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(
            "Sequential Reporter evaluation requires image names like "
            f"<task>_ep<episode>_step<step>.png; got {path!r} at line {line_number}"
        )
    return (
        match.group("task"),
        int(match.group("episode")),
        int(match.group("step")),
    )


def prepare_reporter_sequence(
    samples: Sequence[ReporterSample],
) -> tuple[list[ReporterFrame], int]:
    """Deduplicate and order Reporter rows by task, episode, and current step."""
    frames: list[ReporterFrame] = []
    seen_rows: set[tuple[str, str, str, bool]] = set()
    duplicates_skipped = 0
    for sample in samples:
        message_signature = json.dumps(
            sample.messages,
            sort_keys=True,
            ensure_ascii=False,
        )
        signature = (
            message_signature,
            sample.images[0],
            sample.images[1],
            sample.expected,
        )
        if signature in seen_rows:
            duplicates_skipped += 1
            continue
        seen_rows.add(signature)

        before_task, before_episode, before_step = _parse_reporter_frame_path(
            sample.images[0],
            line_number=sample.line_number,
        )
        current_task, current_episode, current_step = _parse_reporter_frame_path(
            sample.images[1],
            line_number=sample.line_number,
        )
        if (before_task, before_episode) != (current_task, current_episode):
            raise ValueError(
                "Reporter images must come from the same task and episode at "
                f"line {sample.line_number}: {sample.images}"
            )
        if current_step <= before_step:
            raise ValueError(
                "Reporter current frame must be later than its dataset init frame at "
                f"line {sample.line_number}: {before_step} -> {current_step}"
            )
        frames.append(
            ReporterFrame(
                sample=sample,
                task_name=current_task,
                episode_id=current_episode,
                before_step=before_step,
                current_step=current_step,
            )
        )

    frames.sort(
        key=lambda frame: (
            frame.task_name,
            frame.episode_id,
            frame.current_step,
            frame.sample.line_number,
        )
    )
    return frames, duplicates_skipped


def evaluate_reporter_sequence(
    *,
    name: str,
    dataset_path: str | Path,
    engine: ReporterEngine,
    infer_request_type: type,
    request_config: Any,
    model_path: str,
    adapter_path: str | None,
    predictions_path: str | Path,
    image_root: str | Path | None = None,
    max_samples: int | None = None,
    progress_every: int = 50,
) -> ReporterMetrics:
    """Evaluate sequentially while propagating Reporter-predicted init frames.

    The first row of each episode uses its labelled dataset init frame. A
    predicted ``success=true`` promotes the current frame to the init frame for
    the next comparison. False or invalid predictions retain the previous init.
    """
    samples = load_reporter_samples(dataset_path, image_root=image_root)
    frames, duplicates_skipped = prepare_reporter_sequence(samples)
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be at least 1")
        frames = frames[:max_samples]
    if not frames:
        raise ValueError("Reporter test dataset contains no unique sequential samples")

    dataset = Path(dataset_path).expanduser().resolve()
    episode_keys = {(frame.task_name, frame.episode_id) for frame in frames}
    metrics = ReporterMetrics(
        name=name,
        dataset_path=str(dataset),
        model_path=model_path,
        adapter_path=adapter_path or None,
        episodes=len(episode_keys),
        duplicates_skipped=duplicates_skipped,
    )
    output_path = Path(predictions_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    active_episode: tuple[str, int] | None = None
    predicted_init_path: str | None = None
    with output_path.open("w", encoding="utf-8") as output_file:
        for frame in frames:
            sample = frame.sample
            episode_key = (frame.task_name, frame.episode_id)
            episode_started = episode_key != active_episode
            if episode_started:
                active_episode = episode_key
                predicted_init_path = sample.images[0]
            if predicted_init_path is None:  # pragma: no cover - guarded above.
                raise RuntimeError("Reporter sequence has no init frame")

            used_init_path = predicted_init_path
            current_path = sample.images[1]
            request = infer_request_type(
                messages=sample.messages,
                images=[used_init_path, current_path],
            )
            responses = engine.infer([request], request_config=request_config)
            if len(responses) != 1:
                raise RuntimeError(
                    f"Reporter returned {len(responses)} responses for one frame"
                )
            raw_response = _response_text(responses[0])
            predicted = parse_reporter_success(raw_response)
            correct = metrics.update(expected=sample.expected, predicted=predicted)
            init_updated = predicted is True
            if init_updated:
                predicted_init_path = current_path

            output_file.write(
                json.dumps(
                    {
                        "line_number": sample.line_number,
                        "task": frame.task_name,
                        "episode": frame.episode_id,
                        "current_step": frame.current_step,
                        "episode_started": episode_started,
                        "dataset_init_image": sample.images[0],
                        "used_init_image": used_init_path,
                        "current_image": current_path,
                        "init_updated": init_updated,
                        "expected": sample.expected,
                        "predicted": predicted,
                        "parsed": predicted is not None,
                        "correct": correct,
                        "response": raw_response,
                        "messages": sample.messages,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if progress_every > 0 and (
                metrics.total == len(frames) or metrics.total % progress_every == 0
            ):
                accuracy = 100 * metrics.accuracy if metrics.accuracy is not None else 0.0
                print(
                    f"[{name}] {metrics.total}/{len(frames)} frames, "
                    f"correct={metrics.correct}, accuracy={accuracy:.2f}%, "
                    f"invalid={metrics.invalid}"
                )

    return metrics


def evaluate_reporter_dataset(
    *,
    name: str,
    dataset_path: str | Path,
    engine: ReporterEngine,
    infer_request_type: type,
    request_config: Any,
    model_path: str,
    adapter_path: str | None,
    predictions_path: str | Path,
    image_root: str | Path | None = None,
    batch_size: int = 8,
    max_samples: int | None = None,
    progress_every: int = 50,
) -> ReporterMetrics:
    """Run one Reporter against one dataset and persist per-row predictions."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    samples = load_reporter_samples(
        dataset_path,
        image_root=image_root,
        max_samples=max_samples,
    )
    dataset = Path(dataset_path).expanduser().resolve()
    metrics = ReporterMetrics(
        name=name,
        dataset_path=str(dataset),
        model_path=model_path,
        adapter_path=adapter_path or None,
    )
    output_path = Path(predictions_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            requests = [infer_request_type(**sample.as_request()) for sample in batch]
            responses = engine.infer(requests, request_config=request_config)
            if len(responses) != len(batch):
                raise RuntimeError(
                    f"Reporter returned {len(responses)} responses for a batch of {len(batch)} requests"
                )

            for sample, response in zip(batch, responses, strict=True):
                raw_response = _response_text(response)
                predicted = parse_reporter_success(raw_response)
                correct = metrics.update(expected=sample.expected, predicted=predicted)
                output_file.write(
                    json.dumps(
                        {
                            "line_number": sample.line_number,
                            "expected": sample.expected,
                            "predicted": predicted,
                            "parsed": predicted is not None,
                            "correct": correct,
                            "response": raw_response,
                            "messages": sample.messages,
                            "images": sample.images,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if progress_every > 0 and (
                metrics.total == len(samples) or metrics.total % progress_every == 0
            ):
                accuracy = 100 * metrics.accuracy if metrics.accuracy is not None else 0.0
                print(
                    f"[{name}] {metrics.total}/{len(samples)} rows, "
                    f"correct={metrics.correct}, accuracy={accuracy:.2f}%, invalid={metrics.invalid}"
                )

    return metrics


def aggregate_metrics(results: Sequence[ReporterMetrics]) -> dict[str, Any]:
    """Aggregate counts without averaging per-dataset percentages."""
    totals = {
        "total": sum(result.total for result in results),
        "correct": sum(result.correct for result in results),
        "parsed": sum(result.parsed for result in results),
        "invalid": sum(result.invalid for result in results),
        "true_positive": sum(result.true_positive for result in results),
        "true_negative": sum(result.true_negative for result in results),
        "false_positive": sum(result.false_positive for result in results),
        "false_negative": sum(result.false_negative for result in results),
        "completed_total": sum(result.completed_total for result in results),
        "incomplete_total": sum(result.incomplete_total for result in results),
    }
    total = totals["total"]
    parsed = totals["parsed"]
    totals["accuracy"] = totals["correct"] / total if total else None
    totals["valid_accuracy"] = totals["correct"] / parsed if parsed else None
    totals["parse_rate"] = parsed / total if total else None
    totals["completed_recall"] = (
        totals["true_positive"] / totals["completed_total"] if totals["completed_total"] else None
    )
    totals["incomplete_recall"] = (
        totals["true_negative"] / totals["incomplete_total"] if totals["incomplete_total"] else None
    )
    return totals
