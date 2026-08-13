"""Reporter interface for RoboMME evaluation.

The first refactoring step only introduces the role boundary. Reporter-based
subgoal completion decisions will be implemented separately.
"""

from pathlib import Path
from typing import Optional

from env_runner import EnvRunner
from utils import EpisodeState


class ReporterBase:
    def __init__(self, args, save_dir: Path):
        self.args = args
        self.save_dir = save_dir

    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        pass

    def step(self, epstate: EpisodeState, subgoal: Optional[str]) -> None:
        pass

    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        pass


class NullReporter(ReporterBase):
    """No-op Reporter used until a concrete Reporter is configured."""


def build_reporter(args, save_dir: Path) -> ReporterBase:
    if args.reporter_type == "none":
        return NullReporter(args, save_dir)
    raise ValueError(f"Unsupported Reporter type: {args.reporter_type}")
