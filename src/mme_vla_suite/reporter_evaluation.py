"""Offline evaluation utilities for the RoboMME Reporter.

The evaluator consumes the same JSONL format produced by
``build_reporter_dataset.py``.  The final assistant message is treated as the
ground-truth label and is removed before the request is sent to the model.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any
from typing import Protocol

from mme_vla_suite.prompts import DEFAULT_REPORTER_PROMPT_VERSION
from mme_vla_suite.prompts import get_reporter_prompt
from mme_vla_suite.reporter_prompts import DEFAULT_REPORTER_HISTORY_SIZE
from mme_vla_suite.reporter_prompts import build_reporter_image_window
from mme_vla_suite.reporter_prompts import build_reporter_multiview_window
from mme_vla_suite.reporter_prompts import validate_reporter_history_size


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


CONSECUTIVE_TRUE_RETRIGGER_INTERVAL_CALLS = 3


def debounce_reporter_success(
    predicted: bool | None,
    preceding_consecutive_true_count: int,
) -> bool | None:
    """Debounce true runs while allowing a periodic effective retrigger.

    The first true is effective. If true continues uninterrupted, the fourth,
    seventh, and every third later call are effective again. False or invalid
    output is handled by callers as a reset of the consecutive-true counter.
    """
    if predicted is None:
        return None
    if not predicted:
        return False
    return (
        preceding_consecutive_true_count
        % CONSECUTIVE_TRUE_RETRIGGER_INTERVAL_CALLS
        == 0
    )


def _reporter_task_from_messages(messages: Sequence[dict[str, Any]]) -> str:
    """Extract only the subgoal text from a Reporter request."""
    for message in reversed(messages):
        if message.get("role") != "user" or not isinstance(message.get("content"), str):
            continue
        content = message["content"].strip()
        match = re.search(
            r"Current Subgoal:\s*(.*?)\n(?:Observation before executing|Recent )",
            content,
            flags=re.DOTALL,
        )
        return match.group(1).strip() if match else content
    raise ValueError("Reporter request contains no textual user message")


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
    """One labelled Reporter call positioned within an episode."""

    sample: ReporterSample
    task_name: str
    episode_id: int
    current_step: int
    current_image_paths: tuple[str, ...]
    dataset_init_paths: tuple[str, ...] = ()


def score_reporter_completion(
    records: Sequence[dict[str, Any]],
    *,
    early_tolerance_calls: int = 2,
    full_credit_delay_calls: int = 2,
    maximum_delay_calls: int = 4,
    reporter_debounce: bool = True,
) -> dict[str, Any]:
    """Score causal, prefix-safe subgoal completion for every episode.

    A ground-truth ``true`` marks a transition to the next subgoal. With
    ``reporter_debounce`` enabled, predicted true runs keep their first event
    and periodically retrigger after three more uninterrupted true calls. With
    it disabled, every predicted true is effective. Events are consumed in
    order: an event earlier than ``early_tolerance_calls``, a missing
    event, or an event later than ``maximum_delay_calls`` stops progress for
    that episode. Delays beyond ``full_credit_delay_calls`` remain completed
    but are reported as warnings.

    An episode with N expected transitions contains N + 1 subgoals. The final
    subgoal receives credit only when all transitions were safe and no extra
    effective ``true`` occurs afterward.
    """
    if early_tolerance_calls < 0:
        raise ValueError("early_tolerance_calls must be non-negative")
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
        consecutive_true_count = 0
        for call_index, record in enumerate(ordered):
            event = {
                "call_number": call_index + 1,
                "current_step": int(record["current_step"]),
                "line_number": int(record["line_number"]),
            }
            if record.get("expected") is True:
                expected_events.append(event)

            predicted = record.get("predicted")
            predicted_true = predicted is True
            if predicted_true:
                raw_predicted_true_count += 1
            # Sequential inference records already contain the exact debounced
            # decision that drove its init-image/subgoal state. Consume that same
            # value so completion scoring cannot drift from the simulated episode.
            # The fallback keeps this utility usable with minimal hand-built rows.
            effective_true = record.get("effective_true")
            if effective_true is None:
                effective_true = predicted_true
                if reporter_debounce:
                    effective_true = (
                        debounce_reporter_success(predicted, consecutive_true_count)
                        is True
                    )
            if effective_true is True:
                predicted_events.append(event)
            consecutive_true_count = (
                consecutive_true_count + 1 if predicted_true else 0
            )

        transitions: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        predicted_index = 0
        completed_subgoals = 0
        status = "completed"
        failure: dict[str, Any] | None = None
        termination_frame: int | None = None

        def frame_at_call(call_number: int) -> int:
            bounded_call = min(max(call_number, 1), len(ordered))
            return int(ordered[bounded_call - 1]["current_step"])

        for transition_index, expected_event in enumerate(expected_events, start=1):
            if predicted_index >= len(predicted_events):
                status = "stalled_missing_true"
                failure = {
                    "type": status,
                    "subgoal_number": transition_index,
                    "expected_event": expected_event,
                }
                termination_frame = frame_at_call(
                    expected_event["call_number"] + maximum_delay_calls + 1
                )
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

            if delay_calls < -early_tolerance_calls:
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
                termination_frame = predicted_event["current_step"]
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
                termination_frame = frame_at_call(
                    expected_event["call_number"] + maximum_delay_calls + 1
                )
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
                termination_frame = predicted_event["current_step"]
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
                "termination_frame": termination_frame,
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
            "early_tolerance_calls": early_tolerance_calls,
            "full_credit_delay_calls": full_credit_delay_calls,
            "maximum_delay_calls": maximum_delay_calls,
            "early_trigger_policy": "fatal_beyond_tolerance",
            "reporter_debounce": reporter_debounce,
            "consecutive_true_policy": (
                "first_then_periodic_retrigger"
                if reporter_debounce
                else "disabled_raw_predictions"
            ),
            "consecutive_true_retrigger_interval_calls": (
                CONSECUTIVE_TRUE_RETRIGGER_INTERVAL_CALLS
                if reporter_debounce
                else None
            ),
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
    r"^(?P<task>.+)_ep(?P<episode>\d+)_step(?P<step>\d+)"
    r"(?:_(?P<view>overall|wrist))?\.[^.]+$"
)


def _resolve_image_path(raw_path: str, dataset_path: Path, image_root: Path | None) -> Path:
    path = Path(raw_path).expanduser()
    candidates: list[Path] = [path]
    view = path.parent.name if path.parent.name in {"overall", "wrist"} else None
    if image_root is not None:
        if view is not None:
            candidates.append(image_root / view / path.name)
        candidates.extend([image_root / path.name, image_root / raw_path])
    if view is not None:
        candidates.append(dataset_path.parent / "images" / view / path.name)
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
            if not isinstance(images, list) or len(images) < 2:
                raise ValueError(
                    f"Expected at least one Reporter-call observation with its "
                    f"required views at {path}:{line_number}"
                )

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
            image_placeholders = sum(
                str(message.get("content", "")).count("<image>")
                for message in request_messages
                if isinstance(message, dict)
            )
            if image_placeholders != len(images):
                raise ValueError(
                    f"Reporter prompt/image mismatch at {path}:{line_number}: "
                    f"{image_placeholders} placeholders for {len(images)} images"
                )
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


def _parse_reporter_frame_path(
    path: str,
    *,
    line_number: int,
) -> tuple[str, int, int, str | None]:
    match = _REPORTER_FRAME_NAME.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(
            "Sequential Reporter evaluation requires image names like "
            "<task>_ep<episode>_step<step>.png inside an overall/wrist folder "
            "(or a legacy <task>_ep<episode>_step<step>_<view>.png name); "
            f"got {path!r} at line {line_number}"
        )
    view = match.group("view")
    if view is None and Path(path).parent.name in {"overall", "wrist"}:
        view = Path(path).parent.name
    return (
        match.group("task"),
        int(match.group("episode")),
        int(match.group("step")),
        view,
    )


def prepare_reporter_sequence(
    samples: Sequence[ReporterSample],
    reporter_prompt_version: str = DEFAULT_REPORTER_PROMPT_VERSION,
) -> tuple[list[ReporterFrame], int]:
    """Deduplicate and order Reporter rows by task, episode, and current step."""
    reporter_prompt = get_reporter_prompt(reporter_prompt_version)
    frames: list[ReporterFrame] = []
    seen_rows: set[tuple[str, tuple[str, ...], bool]] = set()
    duplicates_skipped = 0
    for sample in samples:
        message_signature = json.dumps(
            sample.messages,
            sort_keys=True,
            ensure_ascii=False,
        )
        signature = (
            message_signature,
            tuple(sample.images),
            sample.expected,
        )
        if signature in seen_rows:
            duplicates_skipped += 1
            continue
        seen_rows.add(signature)

        dataset_init_paths: tuple[str, ...] = ()
        if reporter_prompt.include_init_image:
            init_view_count = len(reporter_prompt.init_image_views)
            dataset_init_paths = tuple(sample.images[:init_view_count])
            parsed_init = [
                _parse_reporter_frame_path(path, line_number=sample.line_number)
                for path in dataset_init_paths
            ]
            before_task, before_episode, before_step, _ = parsed_init[0]
            if any(
                (task, episode, step) != (before_task, before_episode, before_step)
                for task, episode, step, _ in parsed_init
            ):
                raise ValueError(
                    "Reporter init views must share task, episode, and step at "
                    f"line {sample.line_number}: {dataset_init_paths}"
                )
            actual_init_views = tuple(view for _, _, _, view in parsed_init)
            if (
                len(parsed_init) == 1
                and actual_init_views[0]
                not in {None, reporter_prompt.init_image_views[0]}
            ) or (
                len(parsed_init) > 1
                and actual_init_views != reporter_prompt.init_image_views
            ):
                raise ValueError(
                    "Reporter init views must be ordered "
                    f"{reporter_prompt.init_image_views}, got {actual_init_views} "
                    f"at line {sample.line_number}"
                )
            observation_images = sample.images[init_view_count:]
        else:
            before_task = before_episode = before_step = None
            observation_images = sample.images

        view_count = len(reporter_prompt.image_views)
        if len(observation_images) % view_count:
            raise ValueError(
                f"Reporter row at line {sample.line_number} has "
                f"{len(observation_images)} observation images, which is not "
                f"divisible by {view_count} views"
            )
        observation_groups = [
            tuple(observation_images[index : index + view_count])
            for index in range(0, len(observation_images), view_count)
        ]
        if not observation_groups:
            raise ValueError(
                f"Reporter row at line {sample.line_number} contains no "
                "post-init observations"
            )
        parsed_groups = []
        for group in observation_groups:
            parsed = [
                _parse_reporter_frame_path(path, line_number=sample.line_number)
                for path in group
            ]
            task, episode, step, _ = parsed[0]
            if any(
                (item_task, item_episode, item_step) != (task, episode, step)
                for item_task, item_episode, item_step, _ in parsed
            ):
                raise ValueError(
                    "Reporter views for one observation must share task, episode, "
                    f"and step at line {sample.line_number}: {group}"
                )
            actual_views = tuple(view for _, _, _, view in parsed)
            if (
                view_count == 1
                and actual_views[0] not in {None, reporter_prompt.image_views[0]}
            ) or (
                view_count > 1
                and actual_views != reporter_prompt.image_views
            ):
                raise ValueError(
                    "Reporter observation views must be ordered "
                    f"{reporter_prompt.image_views}, got {actual_views} at line "
                    f"{sample.line_number}"
                )
            parsed_groups.append((task, episode, step))

        current_task, current_episode, current_step = parsed_groups[-1]
        if reporter_prompt.include_init_image and any(
            (task, episode) != (before_task, before_episode)
            for task, episode, _ in parsed_groups
        ):
            raise ValueError(
                "Reporter images must come from the same task and episode at "
                f"line {sample.line_number}: {sample.images}"
            )
        if any(
            (task, episode) != (current_task, current_episode)
            for task, episode, _ in parsed_groups
        ):
            raise ValueError(
                "Reporter observations must come from the same task and episode "
                f"at line {sample.line_number}: {sample.images}"
            )
        history_steps = [step for _, _, step in parsed_groups]
        if history_steps != sorted(history_steps):
            raise ValueError(
                "Reporter-call observations must be ordered oldest to newest at "
                f"line {sample.line_number}: {history_steps}"
            )
        if before_step is not None and any(step < before_step for step in history_steps):
            raise ValueError(
                "Reporter-call observations cannot precede the subgoal init frame at "
                f"line {sample.line_number}: {before_step} -> {history_steps}"
            )
        if before_step is not None and current_step <= before_step:
            raise ValueError(
                "Reporter current frame must be later than its dataset init frame at "
                f"line {sample.line_number}: {before_step} -> {current_step}"
            )
        frames.append(
            ReporterFrame(
                sample=sample,
                task_name=current_task,
                episode_id=current_episode,
                current_step=current_step,
                current_image_paths=observation_groups[-1],
                dataset_init_paths=dataset_init_paths,
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


def _prepare_episode_prompt_stages(
    frames: Sequence[ReporterFrame],
) -> tuple[
    dict[tuple[str, int], list[list[dict[str, Any]]]],
    dict[tuple[str, int, int], int],
]:
    """Build ground-truth prompt stages without using them to advance state."""
    grouped: dict[tuple[str, int], list[ReporterFrame]] = {}
    for frame in frames:
        grouped.setdefault((frame.task_name, frame.episode_id), []).append(frame)

    prompt_stages: dict[tuple[str, int], list[list[dict[str, Any]]]] = {}
    dataset_stage_indices: dict[tuple[str, int, int], int] = {}
    for episode_key, episode_frames in grouped.items():
        stages = [episode_frames[0].sample.messages]
        stage_index = 0
        for frame_index, frame in enumerate(episode_frames):
            dataset_stage_indices[
                (frame.task_name, frame.episode_id, frame.sample.line_number)
            ] = stage_index
            if frame.sample.expected and frame_index + 1 < len(episode_frames):
                stage_index += 1
                stages.append(episode_frames[frame_index + 1].sample.messages)
        prompt_stages[episode_key] = stages
    return prompt_stages, dataset_stage_indices


def _format_messages_for_history(
    messages: Sequence[dict[str, Any]],
    subgoal: str,
    observation_count: int,
    reporter_prompt_version: str = DEFAULT_REPORTER_PROMPT_VERSION,
) -> list[dict[str, Any]]:
    """Copy a request and apply the shared prompt at the actual window length."""
    reporter_prompt = get_reporter_prompt(reporter_prompt_version)
    result = [dict(message) for message in messages]
    has_system_message = False
    has_user_message = False
    for message in result:
        if message.get("role") == "system":
            message["content"] = reporter_prompt.system_prompt
            has_system_message = True
        if message.get("role") == "user":
            message["content"] = reporter_prompt.format_user_prompt(
                subgoal,
                observation_count,
            )
            has_user_message = True
    if not has_system_message:
        raise ValueError("Reporter request contains no system message")
    if not has_user_message:
        raise ValueError("Reporter request contains no user message")
    return result


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
    reporter_debounce: bool = True,
    reporter_history_size: int = DEFAULT_REPORTER_HISTORY_SIZE,
    reporter_prompt_version: str = DEFAULT_REPORTER_PROMPT_VERSION,
    prediction_records_out: list[dict[str, Any]] | None = None,
) -> ReporterMetrics:
    """Evaluate with prediction-driven subgoals, paired-view history, and prompts.

    An effective ``success=true`` advances the active subgoal and immediately
    clears the observation window and promotes the current overall/wrist pair
    as the next init pair. The recent window grows naturally from one pair up
    to its capacity and then rolls. With debounce enabled, the first true in a
    run is effective, the next two are suppressed, and the fourth is effective
    again. With it disabled, every parsed true takes effect immediately,
    matching dev_trinity.
    """
    reporter_history_size = validate_reporter_history_size(
        reporter_history_size
    )
    reporter_prompt = get_reporter_prompt(reporter_prompt_version)
    samples = load_reporter_samples(dataset_path, image_root=image_root)
    frames, duplicates_skipped = prepare_reporter_sequence(
        samples,
        reporter_prompt.version,
    )
    init_image_count = (
        len(reporter_prompt.init_image_views)
        if reporter_prompt.include_init_image
        else 0
    )
    view_count = len(reporter_prompt.image_views)
    max_dataset_history = max(
        (len(frame.sample.images) - init_image_count) // view_count
        for frame in frames
    )
    if max_dataset_history > reporter_history_size:
        raise ValueError(
            "Reporter dataset contains a history window of "
            f"{max_dataset_history}, exceeding configured capacity "
            f"{reporter_history_size}"
        )
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be at least 1")
        frames = frames[:max_samples]
    if not frames:
        raise ValueError("Reporter test dataset contains no unique sequential samples")
    prompt_stages, dataset_stage_indices = _prepare_episode_prompt_stages(frames)

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
    predicted_init_paths: tuple[str, ...] = ()
    observation_history: deque[tuple[str, ...]] = deque()
    active_subgoal_index = 0
    consecutive_true_count = 0
    invalid_outputs = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for processed_calls, frame in enumerate(frames, start=1):
            sample = frame.sample
            episode_key = (frame.task_name, frame.episode_id)
            episode_started = episode_key != active_episode
            if episode_started:
                active_episode = episode_key
                predicted_init_paths = frame.dataset_init_paths
                observation_history = deque(maxlen=reporter_history_size)
                active_subgoal_index = 0
                consecutive_true_count = 0
            if reporter_prompt.include_init_image and not predicted_init_paths:
                raise RuntimeError("Reporter sequence has no init frames")

            used_init_paths = predicted_init_paths
            current_paths = frame.current_image_paths
            observation_history.append(current_paths)
            history_count = len(observation_history)
            if reporter_prompt.include_init_image:
                if len(used_init_paths) == 1:
                    used_image_paths = build_reporter_image_window(
                        used_init_paths[0],
                        [paths[0] for paths in observation_history],
                        reporter_history_size,
                    )
                    prompt_observation_count = len(observation_history)
                else:
                    used_image_paths = [
                        *used_init_paths,
                        *build_reporter_multiview_window(
                            observation_history,
                            reporter_history_size,
                        ),
                    ]
                    prompt_observation_count = len(observation_history)
            else:
                used_image_paths = build_reporter_multiview_window(
                    observation_history,
                    reporter_history_size,
                )
                prompt_observation_count = len(observation_history)
            dataset_subgoal_index = dataset_stage_indices[
                (frame.task_name, frame.episode_id, sample.line_number)
            ]
            request_subgoal_index = active_subgoal_index
            stage_messages = prompt_stages[episode_key][request_subgoal_index]
            label_task = _reporter_task_from_messages(sample.messages)
            reporter_task = _reporter_task_from_messages(stage_messages)
            used_messages = _format_messages_for_history(
                stage_messages,
                reporter_task,
                prompt_observation_count,
                reporter_prompt.version,
            )
            reporter_prompt.validate_media_alignment(
                used_messages,
                image_count=len(used_image_paths),
            )
            label_comparable = request_subgoal_index == dataset_subgoal_index
            request = infer_request_type(
                messages=used_messages,
                images=used_image_paths,
            )
            responses = engine.infer([request], request_config=request_config)
            if len(responses) != 1:
                raise RuntimeError(
                    f"Reporter returned {len(responses)} responses for one frame"
                )
            raw_response = _response_text(responses[0])
            predicted = parse_reporter_success(raw_response)
            invalid_outputs += int(predicted is None)
            correct = (
                metrics.update(expected=sample.expected, predicted=predicted)
                if label_comparable
                else None
            )
            if reporter_debounce:
                effective_true = (
                    debounce_reporter_success(predicted, consecutive_true_count)
                    is True
                )
                consecutive_true_count = (
                    consecutive_true_count + 1 if predicted is True else 0
                )
            else:
                # Match dev_trinity: every parsed true immediately takes effect.
                effective_true = predicted is True
                consecutive_true_count = 0
            subgoal_advanced = False
            if effective_true:
                if reporter_prompt.include_init_image:
                    predicted_init_paths = current_paths
                observation_history.clear()
                if active_subgoal_index + 1 < len(prompt_stages[episode_key]):
                    active_subgoal_index += 1
                    subgoal_advanced = True

            internal_record = {
                "line_number": sample.line_number,
                "task": frame.task_name,
                "episode": frame.episode_id,
                "current_step": frame.current_step,
                "episode_started": episode_started,
                "current_image": current_paths[0],
                "current_images": list(current_paths),
                "reporter_history_images": (
                    used_image_paths[len(used_init_paths):]
                    if reporter_prompt.include_init_image
                    else used_image_paths
                ),
                "reporter_history_size": reporter_history_size,
                "reporter_history_count": history_count,
                "reporter_image_views": list(reporter_prompt.image_views),
                "reporter_prompt_version": reporter_prompt.version,
                "reporter_prompt_sha256": reporter_prompt.content_hash,
                "effective_true": effective_true,
                "reporter_debounce": reporter_debounce,
                "subgoal_advanced": subgoal_advanced,
                "active_subgoal_number": request_subgoal_index + 1,
                "next_active_subgoal_number": active_subgoal_index + 1,
                "dataset_subgoal_number": dataset_subgoal_index + 1,
                "label_comparable": label_comparable,
                "expected": sample.expected,
                "predicted": predicted,
                "parsed": predicted is not None,
                "correct": correct,
                "response": raw_response,
                "messages": used_messages,
                "dataset_messages": sample.messages,
                "label_task": label_task,
                "reporter_task": reporter_task,
            }
            if reporter_prompt.include_init_image:
                internal_record.update(
                    {
                        "dataset_init_images": list(frame.dataset_init_paths),
                        "used_init_images": list(used_init_paths),
                        "init_updated": effective_true,
                    }
                )
            if prediction_records_out is not None:
                prediction_records_out.append(internal_record)

            output_record = {
                "task": frame.task_name,
                "episode": frame.episode_id,
                "frame": frame.current_step,
                "current_frames": list(current_paths),
                "reporter_history_images": (
                    used_image_paths[len(used_init_paths):]
                    if reporter_prompt.include_init_image
                    else used_image_paths
                ),
                "reporter_init_images": list(used_init_paths),
                "reporter_image_views": list(reporter_prompt.image_views),
                "reporter_prompt_version": reporter_prompt.version,
                "reporter_prompt_sha256": reporter_prompt.content_hash,
                "expected": sample.expected,
                "predicted": predicted,
                "reporter_history_cleared": effective_true,
                "correct": correct,
                "label_task": label_task,
                "reporter_task": reporter_task,
            }
            output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")

            if progress_every > 0 and (
                processed_calls == len(frames) or processed_calls % progress_every == 0
            ):
                print(
                    f"[{name}] {processed_calls}/{len(frames)} Reporter calls processed, "
                    f"invalid_outputs={invalid_outputs}; completion is scored "
                    "per episode after inference"
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
