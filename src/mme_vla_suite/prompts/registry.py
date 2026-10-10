"""Typed registry for versioned Manager and Reporter prompts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal

from mme_vla_suite.prompts.manager import qwenvl_interaction_verify_v2
from mme_vla_suite.prompts.manager import qwenvl_v1
from mme_vla_suite.prompts.reporter import interaction_aware_v2
from mme_vla_suite.prompts.reporter import temporal_v1


SubgoalType = Literal["simple_subgoal", "grounded_subgoal"]

DEFAULT_REPORTER_HISTORY_SIZE = 7
DEFAULT_REPORTER_PROMPT_VERSION = interaction_aware_v2.VERSION
DEFAULT_MANAGER_PROMPT_VERSION = qwenvl_v1.VERSION


def _content_hash(*parts: str) -> str:
    payload = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _placeholder_count(messages: list[dict[str, str]], placeholder: str) -> int:
    return sum(message.get("content", "").count(placeholder) for message in messages)


@dataclass(frozen=True)
class ReporterPromptSpec:
    """One immutable Reporter system/user prompt pair."""

    version: str
    system_prompt: str
    user_prompt_template: str

    @property
    def content_hash(self) -> str:
        return _content_hash(
            self.system_prompt,
            self.user_prompt_template,
            self.format_user_prompt("<subgoal>", 1),
            self.format_user_prompt(
                "<subgoal>",
                DEFAULT_REPORTER_HISTORY_SIZE,
            ),
        )

    def format_user_prompt(self, subgoal: str, observation_count: int) -> str:
        if observation_count < 1:
            raise ValueError("Reporter history size must be at least 1")
        observation_lines = []
        for index in range(1, observation_count + 1):
            suffix = " (current observation)" if index == observation_count else ""
            observation_lines.append(
                f"Recent observation {index}/{observation_count}{suffix}: <image>"
            )
        return self.user_prompt_template.format(
            subgoal=subgoal,
            observation_lines="\n".join(observation_lines),
        )

    def build_messages(
        self,
        subgoal: str,
        observation_count: int,
    ) -> list[dict[str, str]]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.format_user_prompt(subgoal, observation_count),
            },
        ]
        self.validate_media_alignment(
            messages,
            image_count=observation_count + 1,
        )
        return messages

    @staticmethod
    def validate_media_alignment(
        messages: list[dict[str, str]],
        *,
        image_count: int,
    ) -> None:
        placeholder_count = _placeholder_count(messages, "<image>")
        if placeholder_count != image_count:
            raise ValueError(
                "Reporter prompt/image mismatch: "
                f"{placeholder_count} placeholders for {image_count} images"
            )

    def metadata(self) -> dict[str, str]:
        return {
            "role": "reporter",
            "version": self.version,
            "sha256": self.content_hash,
        }


@dataclass(frozen=True)
class ManagerPromptSpec:
    """One immutable QwenVL Manager prompt family."""

    version: str
    simple_system_prompt: str
    grounded_system_prompt: str
    initial_user_prompt_template: str
    followup_user_prompt_template: str
    reporter_completed_template: str
    reporter_incomplete_template: str
    reporter_missing_template: str

    @property
    def content_hash(self) -> str:
        parts = [
            self.simple_system_prompt,
            self.grounded_system_prompt,
            self.initial_user_prompt_template,
            self.followup_user_prompt_template,
            self.reporter_completed_template,
            self.reporter_incomplete_template,
            self.reporter_missing_template,
        ]
        subgoal_types: tuple[SubgoalType, ...] = (
            "simple_subgoal",
            "grounded_subgoal",
        )
        for subgoal_type in subgoal_types:
            parts.append(
                self.format_user_prompt(
                    subgoal_type=subgoal_type,
                    task_goal="<task_goal>",
                    history_subgoals=[],
                    reporter_result=None,
                    has_video=False,
                )
            )
            for reporter_result in (True, False, None):
                parts.append(
                    self.format_user_prompt(
                        subgoal_type=subgoal_type,
                        task_goal="<task_goal>",
                        history_subgoals=["<subgoal_1>", "<subgoal_2>"],
                        reporter_result=reporter_result,
                        has_video=True,
                    )
                )
        return _content_hash(*parts)

    @staticmethod
    def _subgoal_names(subgoal_type: SubgoalType) -> tuple[str, str]:
        if subgoal_type == "simple_subgoal":
            return "language subgoal", "language subgoals"
        if subgoal_type == "grounded_subgoal":
            return "grounded language subgoal", "grounded language subgoals"
        raise ValueError(
            "Manager subgoal_type must be 'simple_subgoal' or "
            f"'grounded_subgoal', got {subgoal_type!r}"
        )

    def system_prompt(self, subgoal_type: SubgoalType) -> str:
        if subgoal_type == "simple_subgoal":
            return self.simple_system_prompt
        if subgoal_type == "grounded_subgoal":
            return self.grounded_system_prompt
        self._subgoal_names(subgoal_type)
        raise AssertionError("unreachable")

    def format_reporter_result(
        self,
        reporter_result: bool | None,
        subgoal_type: SubgoalType,
    ) -> str:
        subgoal_name, _ = self._subgoal_names(subgoal_type)
        if reporter_result is True:
            template = self.reporter_completed_template
        elif reporter_result is False:
            template = self.reporter_incomplete_template
        else:
            template = self.reporter_missing_template
        return template.format(subgoal_name=subgoal_name)

    def format_user_prompt(
        self,
        *,
        subgoal_type: SubgoalType,
        task_goal: str,
        history_subgoals: list[str],
        reporter_result: bool | None,
        has_video: bool,
    ) -> str:
        subgoal_name, subgoal_name_plural = self._subgoal_names(subgoal_type)
        common = {
            "video_prefix": "<video>" if has_video else "",
            "task_goal": task_goal,
            "subgoal_name": subgoal_name,
            "subgoal_name_plural": subgoal_name_plural,
        }
        if not history_subgoals:
            return self.initial_user_prompt_template.format(**common)
        history = "; ".join(
            f"{index}. {subgoal}"
            for index, subgoal in enumerate(history_subgoals, start=1)
        )
        return self.followup_user_prompt_template.format(
            **common,
            history=history,
            reporter_text=self.format_reporter_result(
                reporter_result,
                subgoal_type,
            ),
        )

    def build_messages(
        self,
        *,
        subgoal_type: SubgoalType,
        task_goal: str,
        history_subgoals: list[str],
        reporter_result: bool | None,
        has_video: bool,
    ) -> list[dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": self.system_prompt(subgoal_type),
            },
            {
                "role": "user",
                "content": self.format_user_prompt(
                    subgoal_type=subgoal_type,
                    task_goal=task_goal,
                    history_subgoals=history_subgoals,
                    reporter_result=reporter_result,
                    has_video=has_video,
                ),
            },
        ]
        self.validate_media_alignment(
            messages,
            image_count=1,
            video_count=int(has_video),
        )
        return messages

    @staticmethod
    def validate_media_alignment(
        messages: list[dict[str, str]],
        *,
        image_count: int,
        video_count: int,
    ) -> None:
        image_placeholders = _placeholder_count(messages, "<image>")
        video_placeholders = _placeholder_count(messages, "<video>")
        if image_placeholders != image_count or video_placeholders != video_count:
            raise ValueError(
                "Manager prompt/media mismatch: "
                f"{image_placeholders} image placeholders for {image_count} images; "
                f"{video_placeholders} video placeholders for {video_count} videos"
            )

    def metadata(self) -> dict[str, str]:
        return {
            "role": "manager_qwenvl",
            "version": self.version,
            "sha256": self.content_hash,
        }


_REPORTER_PROMPTS = {
    temporal_v1.VERSION: ReporterPromptSpec(
        version=temporal_v1.VERSION,
        system_prompt=temporal_v1.SYSTEM_PROMPT,
        user_prompt_template=temporal_v1.USER_PROMPT_TEMPLATE,
    ),
    interaction_aware_v2.VERSION: ReporterPromptSpec(
        version=interaction_aware_v2.VERSION,
        system_prompt=interaction_aware_v2.SYSTEM_PROMPT,
        user_prompt_template=interaction_aware_v2.USER_PROMPT_TEMPLATE,
    ),
}

_REPORTER_ALIASES = {
    "trinity_v2.1": temporal_v1.VERSION,
    "trinity_v2.2": interaction_aware_v2.VERSION,
    "trinity_v2.2-interaction-aware": interaction_aware_v2.VERSION,
}

_MANAGER_PROMPTS = {
    qwenvl_v1.VERSION: ManagerPromptSpec(
        version=qwenvl_v1.VERSION,
        simple_system_prompt=qwenvl_v1.SIMPLE_SYSTEM_PROMPT,
        grounded_system_prompt=qwenvl_v1.GROUNDED_SYSTEM_PROMPT,
        initial_user_prompt_template=qwenvl_v1.INITIAL_USER_PROMPT_TEMPLATE,
        followup_user_prompt_template=qwenvl_v1.FOLLOWUP_USER_PROMPT_TEMPLATE,
        reporter_completed_template=qwenvl_v1.REPORTER_COMPLETED_TEMPLATE,
        reporter_incomplete_template=qwenvl_v1.REPORTER_INCOMPLETE_TEMPLATE,
        reporter_missing_template=qwenvl_v1.REPORTER_MISSING_TEMPLATE,
    ),
    qwenvl_interaction_verify_v2.VERSION: ManagerPromptSpec(
        version=qwenvl_interaction_verify_v2.VERSION,
        simple_system_prompt=qwenvl_interaction_verify_v2.SIMPLE_SYSTEM_PROMPT,
        grounded_system_prompt=qwenvl_interaction_verify_v2.GROUNDED_SYSTEM_PROMPT,
        initial_user_prompt_template=(
            qwenvl_interaction_verify_v2.INITIAL_USER_PROMPT_TEMPLATE
        ),
        followup_user_prompt_template=(
            qwenvl_interaction_verify_v2.FOLLOWUP_USER_PROMPT_TEMPLATE
        ),
        reporter_completed_template=(
            qwenvl_interaction_verify_v2.REPORTER_COMPLETED_TEMPLATE
        ),
        reporter_incomplete_template=(
            qwenvl_interaction_verify_v2.REPORTER_INCOMPLETE_TEMPLATE
        ),
        reporter_missing_template=(
            qwenvl_interaction_verify_v2.REPORTER_MISSING_TEMPLATE
        ),
    ),
}

_MANAGER_ALIASES = {
    "trinity_v2.1": qwenvl_v1.VERSION,
    "trinity_v2.2": qwenvl_v1.VERSION,
}


def available_reporter_prompt_versions() -> tuple[str, ...]:
    return tuple(_REPORTER_PROMPTS)


def available_manager_prompt_versions() -> tuple[str, ...]:
    return tuple(_MANAGER_PROMPTS)


def _resolve_version(
    requested: str,
    registry: dict[str, object],
    aliases: dict[str, str],
    role: str,
) -> str:
    version = aliases.get(requested, requested)
    if version not in registry:
        choices = ", ".join(registry)
        alias_choices = ", ".join(f"{key}->{value}" for key, value in aliases.items())
        raise ValueError(
            f"Unknown {role} prompt version {requested!r}. "
            f"Available versions: {choices}. Aliases: {alias_choices}."
        )
    return version


def get_reporter_prompt(version: str) -> ReporterPromptSpec:
    resolved = _resolve_version(
        version,
        _REPORTER_PROMPTS,
        _REPORTER_ALIASES,
        "Reporter",
    )
    return _REPORTER_PROMPTS[resolved]


def get_manager_prompt(version: str) -> ManagerPromptSpec:
    resolved = _resolve_version(
        version,
        _MANAGER_PROMPTS,
        _MANAGER_ALIASES,
        "Manager",
    )
    return _MANAGER_PROMPTS[resolved]
