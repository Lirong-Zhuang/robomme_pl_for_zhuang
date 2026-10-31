"""Versioned prompt registry shared by dataset builders and evaluation."""

from mme_vla_suite.prompts.registry import DEFAULT_MANAGER_PROMPT_VERSION
from mme_vla_suite.prompts.registry import DEFAULT_REPORTER_PROMPT_VERSION
from mme_vla_suite.prompts.registry import ManagerPromptSpec
from mme_vla_suite.prompts.registry import ReporterPromptSpec
from mme_vla_suite.prompts.registry import available_manager_prompt_versions
from mme_vla_suite.prompts.registry import available_reporter_prompt_versions
from mme_vla_suite.prompts.registry import get_manager_prompt
from mme_vla_suite.prompts.registry import get_reporter_prompt

__all__ = [
    "DEFAULT_MANAGER_PROMPT_VERSION",
    "DEFAULT_REPORTER_PROMPT_VERSION",
    "ManagerPromptSpec",
    "ReporterPromptSpec",
    "available_manager_prompt_versions",
    "available_reporter_prompt_versions",
    "get_manager_prompt",
    "get_reporter_prompt",
]
