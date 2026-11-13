"""Deterministic task-config lookup and per-episode task memory.

The VLM plans with stable ``subgoal_key`` values and ordinary-language subgoals.
This module resolves trusted actions and acceptance criteria from the task config,
keeps a compact ``task_state.json`` for every follow-up call, and appends detailed
events to ``task_events.jsonl`` for audit or recovery.

The implementation assumes one writer per episode directory. It never executes an
action by name; callers must resolve the returned action name through their own
trusted action registry.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import string
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any


_ACCEPTANCE_STATES = {"completed", "incomplete", "failed"}
_FOLLOWUP_STATUSES = _ACCEPTANCE_STATES
_SUBGOAL_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_STOP_KEY = "STOP"
_STOP_TEXT = "STOP"
_PLANNER_SUBGOAL_FIELDS = {"subgoal_key", "subgoal", "arguments"}
_FIRST_PLAN_FIELDS = {
    "subgoal_sequence",
    "next_subgoal_key",
    "next_subgoal",
    "arguments",
}
_FOLLOWUP_FIELDS = {
    "status",
    "description",
    "next_subgoal_key",
    "next_subgoal",
    "arguments",
}
_KEYFRAME_SAVE_FIELD = "keyframe_save"


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _required_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _json_copy(value: Any) -> Any:
    """Return a JSON-compatible deep copy and reject non-JSON values."""

    return json.loads(json.dumps(value, ensure_ascii=False))


def _config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ResolvedSubgoal:
    """One planner subgoal resolved against the trusted task config."""

    subgoal_key: str
    subgoal: str
    arguments: dict[str, Any]
    action_name: str
    acceptance: dict[str, str]

    def planner_payload(self) -> dict[str, Any]:
        return {
            "subgoal_key": self.subgoal_key,
            "subgoal": self.subgoal,
            "arguments": _json_copy(self.arguments),
        }

    def execution_payload(self) -> dict[str, Any]:
        return {
            **self.planner_payload(),
            "action_name": self.action_name,
            "acceptance": dict(self.acceptance),
        }


class SubgoalResolver:
    """Resolve VLM subgoal keys to trusted actions and acceptance criteria."""

    def __init__(
        self,
        task_config: Mapping[str, Any],
        allowed_action_names: set[str],
    ) -> None:
        if not allowed_action_names:
            raise ValueError("allowed_action_names must not be empty")

        subgoals = task_config.get("subgoals")
        if not isinstance(subgoals, list) or not subgoals:
            raise ValueError("task_config.subgoals must be a non-empty list")

        self._by_key: dict[str, dict[str, Any]] = {}
        for index, raw_spec in enumerate(subgoals):
            if not isinstance(raw_spec, Mapping):
                raise ValueError(f"subgoals[{index}] must be an object")

            key = _required_string(
                f"subgoals[{index}].subgoal_key",
                raw_spec.get("subgoal_key"),
            )
            if not _SUBGOAL_KEY_PATTERN.fullmatch(key):
                raise ValueError(
                    f"subgoals[{index}].subgoal_key has invalid format: {key!r}"
                )
            if key in self._by_key:
                raise ValueError(f"duplicate subgoal_key: {key!r}")

            template = _required_string(
                f"subgoals[{index}].template",
                raw_spec.get("template"),
            )
            action_name = _required_string(
                f"subgoals[{index}].action_name",
                raw_spec.get("action_name"),
            )
            if action_name not in allowed_action_names:
                raise ValueError(
                    f"subgoal {key!r} uses untrusted action {action_name!r}"
                )

            acceptance = raw_spec.get("acceptance")
            if not isinstance(acceptance, Mapping) or set(acceptance) != _ACCEPTANCE_STATES:
                raise ValueError(
                    f"subgoal {key!r} acceptance must define exactly "
                    "completed, incomplete, and failed"
                )
            normalized_acceptance = {
                state: _required_string(
                    f"subgoals[{index}].acceptance.{state}",
                    acceptance[state],
                )
                for state in sorted(_ACCEPTANCE_STATES)
            }
            argument_names = _template_fields(
                template,
                *normalized_acceptance.values(),
            )

            self._by_key[key] = {
                "subgoal_key": key,
                "template": template,
                "action_name": action_name,
                "acceptance": normalized_acceptance,
                "argument_names": argument_names,
            }

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._by_key)

    def resolve(
        self,
        *,
        subgoal_key: str,
        subgoal: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> ResolvedSubgoal:
        key = _required_string("subgoal_key", subgoal_key)
        text = _required_string("subgoal", subgoal)
        if key == _STOP_KEY:
            raise ValueError("STOP is not an executable subgoal_key")

        try:
            spec = self._by_key[key]
        except KeyError as error:
            raise ValueError(f"unknown subgoal_key: {key!r}") from error

        normalized_arguments = dict(arguments or {})
        expected_arguments = spec["argument_names"]
        actual_arguments = set(normalized_arguments)
        if actual_arguments != expected_arguments:
            missing = sorted(expected_arguments - actual_arguments)
            extra = sorted(actual_arguments - expected_arguments)
            raise ValueError(
                f"arguments mismatch for subgoal_key {key!r}; "
                f"missing arguments={missing}, unexpected arguments={extra}"
            )
        try:
            expected_text = spec["template"].format_map(normalized_arguments)
            acceptance = {
                state: criterion.format_map(normalized_arguments)
                for state, criterion in spec["acceptance"].items()
            }
        except KeyError as error:
            missing = error.args[0]
            raise ValueError(
                f"missing argument {missing!r} for subgoal_key {key!r}"
            ) from error

        if _normalized_text(text) != _normalized_text(expected_text):
            raise ValueError(
                f"subgoal text does not match key {key!r}: "
                f"expected {expected_text!r}, got {text!r}"
            )

        return ResolvedSubgoal(
            subgoal_key=key,
            subgoal=expected_text,
            arguments=_json_copy(normalized_arguments),
            action_name=spec["action_name"],
            acceptance=acceptance,
        )

    def resolve_planner_payload(self, payload: Mapping[str, Any]) -> ResolvedSubgoal:
        if not isinstance(payload, Mapping):
            raise ValueError("planner subgoal must be an object")
        if set(payload) != _PLANNER_SUBGOAL_FIELDS:
            raise ValueError(
                "planner subgoal must contain exactly "
                "subgoal_key, subgoal, and arguments"
            )
        arguments = payload.get("arguments", {})
        if not isinstance(arguments, Mapping):
            raise ValueError("planner subgoal arguments must be an object")
        return self.resolve(
            subgoal_key=payload.get("subgoal_key"),
            subgoal=payload.get("subgoal"),
            arguments=arguments,
        )


class TaskStateStore:
    """Read and atomically replace the compact per-episode state file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"task state must contain one object: {self.path}")
        return value

    def save(self, state: Mapping[str, Any]) -> None:
        payload = json.dumps(
            state,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        temporary_path = self.path.with_name(
            f".{self.path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary_path.open("w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


class TaskEventLog:
    """Append and query detailed JSONL events for audit and recovery."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._next_event_id = self._find_next_event_id()

    def _find_next_event_id(self) -> int:
        if not self.path.exists():
            return 1
        last_id = 0
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                event = json.loads(line)
                event_id = event.get("event_id")
                if not isinstance(event_id, int) or event_id <= last_id:
                    raise ValueError(
                        f"invalid event_id at {self.path}:{line_number}"
                    )
                last_id = event_id
        return last_id + 1

    def append(self, event_type: str, **fields: Any) -> dict[str, Any]:
        event = {
            "event_id": self._next_event_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "type": _required_string("event_type", event_type),
            **_json_copy(fields),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._next_event_id += 1
        return event

    def query(
        self,
        *,
        event_type: str | None = None,
        subgoal_id: str | None = None,
        after_event_id: int = 0,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        if after_event_id < 0:
            raise ValueError("after_event_id must be non-negative")
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")

        matches: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event["event_id"] <= after_event_id:
                    continue
                if event_type is not None and event.get("type") != event_type:
                    continue
                if subgoal_id is not None and event.get("subgoal_id") != subgoal_id:
                    continue
                matches.append(event)
                if limit is not None and len(matches) >= limit:
                    break
        return matches


class TaskMemory:
    """High-level task state, event log, and subgoal-resolution API."""

    def __init__(
        self,
        episode_dir: Path,
        task_config: Mapping[str, Any],
        allowed_action_names: set[str],
    ) -> None:
        self.episode_dir = Path(episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.task_config = _json_copy(task_config)
        self.task_config_hash = _config_hash(task_config)
        planning = self.task_config.get("planning", {})
        if planning and not isinstance(planning, Mapping):
            raise ValueError("task_config.planning must be an object")
        information_sources = planning.get("information_sources", [])
        if information_sources and not isinstance(information_sources, list):
            raise ValueError("planning.information_sources must be a list")
        self.information_sources = tuple(information_sources)
        keyframe_policies = self.task_config.get("keyframes", [])
        if not isinstance(keyframe_policies, list):
            raise ValueError("task_config.keyframes must be a list")
        if any(not isinstance(policy, Mapping) for policy in keyframe_policies):
            raise ValueError("every task_config keyframe policy must be an object")
        for index, policy in enumerate(keyframe_policies):
            capture_mode = _required_string(
                f"keyframes[{index}].capture_mode",
                policy.get("capture_mode"),
            )
            if capture_mode not in {"single", "sequence"}:
                raise ValueError(
                    f"keyframes[{index}].capture_mode must be single or sequence"
                )
            frame_count = policy.get("frame_count")
            if not isinstance(frame_count, int) or isinstance(frame_count, bool):
                raise ValueError(f"keyframes[{index}].frame_count must be an integer")
            if capture_mode == "single" and frame_count != 1:
                raise ValueError(
                    f"keyframes[{index}].single capture requires frame_count=1"
                )
            if capture_mode == "sequence" and frame_count <= 1:
                raise ValueError(
                    f"keyframes[{index}].sequence capture requires frame_count>1"
                )
            if capture_mode == "sequence":
                _required_string(
                    f"keyframes[{index}].stop_when",
                    policy.get("stop_when"),
                )
            if policy.get("save_description") is not True:
                raise ValueError(
                    f"keyframes[{index}].save_description must be true"
                )
        self.keyframe_policies = keyframe_policies
        self.resolver = SubgoalResolver(task_config, allowed_action_names)
        self.state_store = TaskStateStore(self.episode_dir / "task_state.json")
        self.event_log = TaskEventLog(self.episode_dir / "task_events.jsonl")

    @classmethod
    def from_paths(
        cls,
        *,
        episode_dir: Path,
        task_config_path: Path,
        action_catalog_path: Path,
    ) -> TaskMemory:
        task_config = _load_json_object(task_config_path)
        action_catalog = _load_json_object(action_catalog_path)
        actions = action_catalog.get("actions")
        if not isinstance(actions, list):
            raise ValueError("action catalog must contain an actions list")
        allowed_action_names = {
            action["name"]
            for action in actions
            if isinstance(action, Mapping) and isinstance(action.get("name"), str)
        }
        return cls(episode_dir, task_config, allowed_action_names)

    def load_state(self) -> dict[str, Any]:
        return self.state_store.load()

    @property
    def uses_keyframes(self) -> bool:
        return bool(self.keyframe_policies)

    @property
    def uses_video_subgoal_list(self) -> bool:
        return "video_subgoal_list" in self.information_sources

    def build_first_planning_inputs(
        self,
        *,
        task_goal: str,
        current_observations: Any,
        keyframe_policy_index: int = 0,
    ) -> dict[str, Any]:
        """Build template values for the first in-task planning call."""

        inputs = {
            "task_goal": _required_string("task_goal", task_goal),
            "current_observations": _json_copy(current_observations),
        }
        if self.uses_keyframes:
            inputs["keyframe_policy"] = self._keyframe_policy_input(
                keyframe_policy_index
            )
            inputs["saved_keyframe"] = None
        return inputs

    def commit_first_plan(
        self,
        *,
        task_goal: str,
        planner_output: Mapping[str, Any],
        persistent_facts: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.load_state():
            raise RuntimeError("first plan cannot be committed after state exists")
        expected_fields = self._expected_output_fields(_FIRST_PLAN_FIELDS)
        if set(planner_output) != expected_fields:
            suffix = ", and keyframe_save" if self.uses_keyframes else ""
            raise ValueError(
                "first planner output must contain exactly subgoal_sequence, "
                f"next_subgoal_key, next_subgoal, arguments{suffix}"
            )
        keyframe_save = self._keyframe_selection(planner_output)

        raw_sequence = planner_output.get("subgoal_sequence")
        if not isinstance(raw_sequence, list) or not raw_sequence:
            raise ValueError("subgoal_sequence must be a non-empty list")
        resolved_sequence = [
            self.resolver.resolve_planner_payload(item) for item in raw_sequence
        ]

        next_resolved = self._resolve_next_output(planner_output)
        if next_resolved is None:
            raise ValueError("first plan cannot start with STOP")
        if next_resolved.planner_payload() != resolved_sequence[0].planner_payload():
            raise ValueError("first next_subgoal must equal subgoal_sequence[0]")

        sequence = [
            self._new_instance(resolved, number=index + 1)
            for index, resolved in enumerate(resolved_sequence)
        ]
        plan_id = "plan-0001"
        event = self.event_log.append(
            "plan_created",
            plan_id=plan_id,
            task_goal=_required_string("task_goal", task_goal),
            subgoal_sequence=sequence,
            keyframe_save=keyframe_save,
        )
        state = {
            "schema_version": "task_state.v1",
            "task_config": {
                "schema_version": self.task_config.get("schema_version"),
                "sha256": self.task_config_hash,
            },
            "task_goal": task_goal,
            "plan": {
                "plan_id": plan_id,
                "sequence": sequence,
                "current_index": 0,
                "next_instance_number": len(sequence) + 1,
            },
            "current_subgoal": sequence[0],
            "progress": {
                "completed_subgoal_ids": [],
            },
            "persistent_facts": _json_copy(dict(persistent_facts or {})),
            "last_action": None,
            "last_verification": None,
            "event_cursor": event["event_id"],
        }
        self.state_store.save(state)
        return _json_copy(state)

    def current_execution(self) -> dict[str, Any] | None:
        state = self._required_state()
        current = state.get("current_subgoal")
        if current is None:
            return None
        resolved = self.resolver.resolve(
            subgoal_key=current["subgoal_key"],
            subgoal=current["subgoal"],
            arguments=current.get("arguments", {}),
        )
        return {
            "instance_id": current["instance_id"],
            "attempt": current["attempt"],
            **resolved.execution_payload(),
        }

    def resolve_current_action(
        self,
        action_registry: Mapping[str, Any],
    ) -> Any | None:
        """Return the trusted callable for the current key, or ``None`` at STOP."""

        execution = self.current_execution()
        if execution is None:
            return None
        action_name = execution["action_name"]
        try:
            action = action_registry[action_name]
        except KeyError as error:
            raise RuntimeError(
                f"action registry has no implementation for {action_name!r}"
            ) from error
        if not callable(action):
            raise TypeError(f"registered action {action_name!r} is not callable")
        return action

    def record_action_started(self, **metadata: Any) -> dict[str, Any]:
        state = self._required_state()
        execution = self.current_execution()
        if execution is None:
            raise RuntimeError("cannot start an action after STOP")
        event = self.event_log.append(
            "action_started",
            subgoal_id=execution["instance_id"],
            subgoal_key=execution["subgoal_key"],
            action_name=execution["action_name"],
            attempt=execution["attempt"],
            metadata=metadata,
        )
        state["last_action"] = {
            "status": "started",
            "event_id": event["event_id"],
            "subgoal_id": execution["instance_id"],
            "action_name": execution["action_name"],
        }
        state["event_cursor"] = event["event_id"]
        self.state_store.save(state)
        return event

    def record_action_finished(
        self,
        *,
        runtime_status: str,
        artifact_refs: list[str] | None = None,
        **metadata: Any,
    ) -> dict[str, Any]:
        state = self._required_state()
        execution = self.current_execution()
        if execution is None:
            raise RuntimeError("cannot finish an action after STOP")
        event = self.event_log.append(
            "action_finished",
            subgoal_id=execution["instance_id"],
            subgoal_key=execution["subgoal_key"],
            action_name=execution["action_name"],
            attempt=execution["attempt"],
            runtime_status=_required_string("runtime_status", runtime_status),
            artifact_refs=list(artifact_refs or []),
            metadata=metadata,
        )
        state["last_action"] = {
            "status": "finished",
            "event_id": event["event_id"],
            "subgoal_id": execution["instance_id"],
            "action_name": execution["action_name"],
            "runtime_status": runtime_status,
        }
        state["event_cursor"] = event["event_id"]
        self.state_store.save(state)
        return event

    def commit_followup(
        self,
        planner_output: Mapping[str, Any],
    ) -> dict[str, Any]:
        state = self._required_state()
        current = state.get("current_subgoal")
        if current is None:
            raise RuntimeError("cannot commit follow-up after STOP")
        expected_fields = self._expected_output_fields(_FOLLOWUP_FIELDS)
        if set(planner_output) != expected_fields:
            suffix = ", and keyframe_save" if self.uses_keyframes else ""
            raise ValueError(
                "follow-up output must contain exactly status, description, "
                f"next_subgoal_key, next_subgoal, arguments{suffix}"
            )
        keyframe_save = self._keyframe_selection(planner_output)

        status = _required_string("status", planner_output.get("status"))
        if status not in _FOLLOWUP_STATUSES:
            raise ValueError(f"invalid follow-up status: {status!r}")
        description = _required_string(
            "description",
            planner_output.get("description"),
        )
        next_resolved = self._resolve_next_output(planner_output)
        current_resolved = self.resolver.resolve(
            subgoal_key=current["subgoal_key"],
            subgoal=current["subgoal"],
            arguments=current.get("arguments", {}),
        )

        if status == "incomplete":
            if next_resolved is None:
                raise ValueError("incomplete status cannot transition to STOP")
            if next_resolved.planner_payload() != current_resolved.planner_payload():
                raise ValueError(
                    "incomplete status must retain the current subgoal and arguments"
                )

        next_instance = self._transition_state(
            state=state,
            status=status,
            current=current,
            next_resolved=next_resolved,
        )
        event = self.event_log.append(
            "verification_and_transition",
            subgoal_id=current["instance_id"],
            subgoal_key=current["subgoal_key"],
            status=status,
            description=description,
            next_subgoal=next_instance,
            keyframe_save=keyframe_save,
        )
        state["last_verification"] = {
            "event_id": event["event_id"],
            "subgoal_id": current["instance_id"],
            "status": status,
            "description": description,
        }
        state["event_cursor"] = event["event_id"]
        self.state_store.save(state)
        return _json_copy(state)

    def build_followup_inputs(
        self,
        *,
        verification_parameters: Mapping[str, Any],
        subgoal_start_observations: Any,
        recent_observations: Any,
        current_observation: Any | None = None,
        keyframe_policy_index: int = 0,
    ) -> dict[str, Any]:
        state = self._required_state()
        execution = self.current_execution()
        if execution is None:
            raise RuntimeError("cannot build follow-up inputs after STOP")
        inputs = {
            "task_goal": state["task_goal"],
            "task_state": _json_copy(state),
            "current_subgoal_key": execution["subgoal_key"],
            "current_subgoal": execution["subgoal"],
            "acceptance_criteria": dict(execution["acceptance"]),
            "verification_parameters": _json_copy(verification_parameters),
            "subgoal_start_observations": _json_copy(subgoal_start_observations),
            "recent_observations": _json_copy(recent_observations),
            "current_subgoal_sequence": _json_copy(state["plan"]["sequence"]),
        }
        if self.uses_keyframes:
            if current_observation is None:
                raise ValueError(
                    "keyframe tasks require a separately identified current_observation"
                )
            inputs["current_observation"] = _json_copy(current_observation)
            inputs["keyframe_policy"] = self._keyframe_policy_input(
                keyframe_policy_index
            )
            inputs["saved_keyframe"] = self._saved_keyframe_input(
                state,
                keyframe_policy_index,
            )
        if self.uses_video_subgoal_list:
            inputs["video_subgoal_list"] = [
                {
                    "subgoal_key": item["subgoal_key"],
                    "subgoal": item["subgoal"],
                    "arguments": _json_copy(item.get("arguments", {})),
                }
                for item in state["plan"]["sequence"]
            ]
        return inputs

    def save_keyframe_selection(
        self,
        *,
        keyframe_save: str | None,
        observation_paths: Mapping[str, str],
        description: str | None = None,
        policy_index: int = 0,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Resolve one VLM observation ID and retain its trusted image address.

        The VLM returns only an ID from the observation candidates supplied by the
        execution loop. The loop later calls this method with its trusted ID-to-path
        mapping. This method stores the resolved reference in the configured compact
        task-state field; it does not copy or move the image file itself.
        """

        state = self._required_state()
        if keyframe_save is None:
            return _json_copy(state)
        if not self.uses_keyframes:
            raise ValueError("task config does not enable keyframe saving")
        observation_id = _required_string("keyframe_save", keyframe_save)
        if not isinstance(observation_paths, Mapping):
            raise ValueError("observation_paths must be an ID-to-path object")
        try:
            image_path = _required_string(
                f"observation_paths[{observation_id!r}]",
                observation_paths[observation_id],
            )
        except KeyError as error:
            raise ValueError(
                f"keyframe_save is not a supplied observation ID: {observation_id!r}"
            ) from error
        if not isinstance(policy_index, int) or not (
            0 <= policy_index < len(self.keyframe_policies)
        ):
            raise ValueError(
                "policy_index does not identify a configured keyframe policy"
            )

        policy = self.keyframe_policies[policy_index]
        field_path = _required_string(
            f"keyframes[{policy_index}].task_state_field",
            policy.get("task_state_field"),
        )
        if description is None:
            raise ValueError(
                "description is required for every saved keyframe observation"
            )
        record = {
            "observation_id": observation_id,
            "image_path": image_path,
            "description": _required_string("description", description),
            "policy_index": policy_index,
        }

        existing = self._task_state_field_value(state, field_path)
        capture_mode = policy["capture_mode"]
        if capture_mode == "sequence":
            frame_count = policy["frame_count"]
            if overwrite:
                records: list[dict[str, Any]] = []
            elif existing is None:
                records = []
            elif isinstance(existing, list):
                records = existing
            else:
                raise ValueError(
                    f"saved keyframe sequence must be a list: {field_path}"
                )
            for saved_record in records:
                if saved_record.get("observation_id") == observation_id:
                    if saved_record == record:
                        return _json_copy(state)
                    raise RuntimeError(
                        "keyframe observation ID is already saved with different "
                        f"content: {observation_id!r}"
                    )
            if len(records) >= frame_count:
                raise RuntimeError(
                    "keyframe sequence already reached the frame_count upper "
                    f"bound of {frame_count}: "
                    f"{field_path}"
                )
            records.append(record)
            return self.set_task_state_field(
                field_path,
                records,
                source=f"keyframe_observation:{observation_id}",
            )

        if existing is not None and not overwrite:
            if existing == record:
                return _json_copy(state)
            raise RuntimeError(
                f"single keyframe task-state field is already populated: {field_path}"
            )
        return self.set_task_state_field(
            field_path,
            record,
            source=f"keyframe_observation:{observation_id}",
        )

    def update_persistent_facts(
        self,
        facts: Mapping[str, Any],
    ) -> dict[str, Any]:
        state = self._required_state()
        state["persistent_facts"].update(_json_copy(dict(facts)))
        event = self.event_log.append(
            "persistent_facts_updated",
            facts=dict(facts),
        )
        state["event_cursor"] = event["event_id"]
        self.state_store.save(state)
        return _json_copy(state)

    def set_task_state_field(
        self,
        field_path: str,
        value: Any,
        *,
        source: str,
    ) -> dict[str, Any]:
        """Set one configured dotted path under ``persistent_facts``.

        Keyframe and video-derived policies use this method for the compact value
        needed by normal follow-up calls. The same update is appended to the event
        log so the full provenance remains available for recovery.
        """

        path = _required_string("field_path", field_path)
        parts = path.split(".")
        if len(parts) < 2 or parts[0] != "persistent_facts":
            raise ValueError(
                "task-state field path must start with 'persistent_facts.'"
            )
        if any(not part or not _SUBGOAL_KEY_PATTERN.fullmatch(part) for part in parts[1:]):
            raise ValueError(f"invalid task-state field path: {field_path!r}")

        state = self._required_state()
        target = state["persistent_facts"]
        for part in parts[1:-1]:
            child = target.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(
                    f"cannot write through non-object task-state field {part!r}"
                )
            target = child
        target[parts[-1]] = _json_copy(value)

        event = self.event_log.append(
            "task_state_field_updated",
            field_path=path,
            value=value,
            source=_required_string("source", source),
        )
        state["event_cursor"] = event["event_id"]
        self.state_store.save(state)
        return _json_copy(state)

    def query_events(self, **filters: Any) -> list[dict[str, Any]]:
        return self.event_log.query(**filters)

    def _expected_output_fields(self, base_fields: set[str]) -> set[str]:
        fields = set(base_fields)
        if self.uses_keyframes:
            fields.add(_KEYFRAME_SAVE_FIELD)
        return fields

    def _keyframe_policy_input(self, policy_index: int) -> dict[str, Any]:
        if not isinstance(policy_index, int) or not (
            0 <= policy_index < len(self.keyframe_policies)
        ):
            raise ValueError("keyframe policy index is out of range")
        policy = _json_copy(self.keyframe_policies[policy_index])
        policy["task_config_ref"] = f"task_config.keyframes[{policy_index}]"
        return policy

    def _saved_keyframe_input(
        self,
        state: Mapping[str, Any],
        policy_index: int,
    ) -> dict[str, Any] | None:
        policy = self._keyframe_policy_input(policy_index)
        field_path = _required_string(
            f"keyframes[{policy_index}].task_state_field",
            policy.get("task_state_field"),
        )
        saved = self._task_state_field_value(state, field_path)
        if saved is None:
            return None
        if policy["capture_mode"] == "sequence":
            if not isinstance(saved, list):
                raise ValueError(
                    f"saved keyframe sequence must be a list: {field_path}"
                )
            observations = []
            for index, item in enumerate(saved):
                if not isinstance(item, Mapping):
                    raise ValueError(
                        f"saved keyframe sequence item must be an object: "
                        f"{field_path}[{index}]"
                    )
                observations.append(
                    {
                        "observation_id": item.get("observation_id"),
                        "image": "<image>",
                        "image_path": _required_string(
                            "saved keyframe image_path",
                            item.get("image_path"),
                        ),
                        "description": item.get("description"),
                    }
                )
            frame_count = policy["frame_count"]
            return {
                "task_config_ref": f"task_config.keyframes[{policy_index}]",
                "task_state_ref": field_path,
                "capture_mode": "sequence",
                "saved_count": len(observations),
                "frame_count_upper_bound": frame_count,
                "capacity_reached": len(observations) >= frame_count,
                "stop_when": policy["stop_when"],
                "observations": observations,
            }
        if not isinstance(saved, Mapping):
            raise ValueError(f"saved keyframe must be an object: {field_path}")
        image_path = _required_string(
            "saved keyframe image_path",
            saved.get("image_path"),
        )
        return {
            "task_config_ref": f"task_config.keyframes[{policy_index}]",
            "task_state_ref": field_path,
            "observation_id": saved.get("observation_id"),
            "image": "<image>",
            "image_path": image_path,
            "description": saved.get("description"),
        }

    def _keyframe_selection(self, planner_output: Mapping[str, Any]) -> str | None:
        if not self.uses_keyframes:
            return None
        value = planner_output.get(_KEYFRAME_SAVE_FIELD)
        if value is None:
            return None
        return _required_string(_KEYFRAME_SAVE_FIELD, value)

    @staticmethod
    def _task_state_field_value(state: Mapping[str, Any], field_path: str) -> Any:
        parts = field_path.split(".")
        if len(parts) < 2 or parts[0] != "persistent_facts":
            raise ValueError(
                "task-state field path must start with 'persistent_facts.'"
            )
        value: Any = state
        for part in parts:
            if not isinstance(value, Mapping) or part not in value:
                return None
            value = value[part]
        return _json_copy(value)

    def _required_state(self) -> dict[str, Any]:
        state = self.load_state()
        if not state:
            raise RuntimeError("task state does not exist; commit the first plan first")
        if state.get("task_config", {}).get("sha256") != self.task_config_hash:
            raise RuntimeError("task state was created from a different task config")
        return state

    def _resolve_next_output(
        self,
        planner_output: Mapping[str, Any],
    ) -> ResolvedSubgoal | None:
        key = _required_string(
            "next_subgoal_key",
            planner_output.get("next_subgoal_key"),
        )
        text = _required_string("next_subgoal", planner_output.get("next_subgoal"))
        arguments = planner_output.get("arguments", {})
        if not isinstance(arguments, Mapping):
            raise ValueError("arguments must be an object")
        if key == _STOP_KEY:
            if text != _STOP_TEXT or arguments:
                raise ValueError("STOP requires next_subgoal='STOP' and empty arguments")
            return None
        return self.resolver.resolve(
            subgoal_key=key,
            subgoal=text,
            arguments=arguments,
        )

    def _transition_state(
        self,
        *,
        state: dict[str, Any],
        status: str,
        current: dict[str, Any],
        next_resolved: ResolvedSubgoal | None,
    ) -> dict[str, Any] | None:
        plan = state["plan"]
        current_index = plan["current_index"]

        if status == "incomplete":
            current["attempt"] += 1
            plan["sequence"][current_index] = current
            state["current_subgoal"] = current
            return _json_copy(current)

        if status == "completed":
            completed_ids = state["progress"]["completed_subgoal_ids"]
            if current["instance_id"] not in completed_ids:
                completed_ids.append(current["instance_id"])

        if next_resolved is None:
            state["current_subgoal"] = None
            plan["current_index"] = len(plan["sequence"])
            return None

        if status == "failed" and self._same_subgoal(current, next_resolved):
            current["attempt"] += 1
            plan["sequence"][current_index] = current
            state["current_subgoal"] = current
            return _json_copy(current)

        candidate_index = current_index + 1
        if candidate_index < len(plan["sequence"]):
            candidate = plan["sequence"][candidate_index]
            if self._same_subgoal(candidate, next_resolved):
                plan["current_index"] = candidate_index
                state["current_subgoal"] = candidate
                return _json_copy(candidate)

        number = plan["next_instance_number"]
        next_instance = self._new_instance(next_resolved, number=number)
        plan["next_instance_number"] = number + 1
        plan["sequence"].append(next_instance)
        plan["current_index"] = len(plan["sequence"]) - 1
        state["current_subgoal"] = next_instance
        return _json_copy(next_instance)

    @staticmethod
    def _new_instance(
        resolved: ResolvedSubgoal,
        *,
        number: int,
    ) -> dict[str, Any]:
        return {
            "instance_id": f"sg-{number:04d}",
            **resolved.planner_payload(),
            "attempt": 1,
        }

    @staticmethod
    def _same_subgoal(
        instance: Mapping[str, Any],
        resolved: ResolvedSubgoal,
    ) -> bool:
        return (
            instance.get("subgoal_key") == resolved.subgoal_key
            and _normalized_text(str(instance.get("subgoal", "")))
            == _normalized_text(resolved.subgoal)
            and instance.get("arguments", {}) == resolved.arguments
        )


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON resource must contain one object: {path}")
    return value


def _template_fields(*templates: str) -> set[str]:
    fields: set[str] = set()
    formatter = string.Formatter()
    for template in templates:
        for _, field_name, _, _ in formatter.parse(template):
            if field_name is None:
                continue
            if _SUBGOAL_KEY_PATTERN.fullmatch(field_name) is None:
                raise ValueError(
                    f"template placeholder must be a simple snake_case name: "
                    f"{field_name!r}"
                )
            fields.add(field_name)
    return fields

