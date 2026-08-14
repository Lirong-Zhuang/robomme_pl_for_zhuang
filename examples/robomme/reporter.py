"""Reporter implementations for RoboMME evaluation."""

import json
import pprint
import shutil
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
from swift.llm import InferRequest, PtEngine, RequestConfig

from env_runner import EnvRunner
from utils import EpisodeState


REPORTER_SYSTEM_PROMPT = (
    "You are a helpful assistant to determine whether the current robot subgoal "
    "is complete by comparing two observations. "
    'Return only {"success": true} or {"success": false}. '
)

REPORTER_USER_PROMPT = (
    "Subgoal: {subgoal}\n"
    "Observation before executing the subgoal: <image>\n"
    "Observation after execution: <image>\n"
    "Determine whether the subgoal is complete based on the observations before and after execution. "
)


class ReporterBase:
    def __init__(self, args, save_dir: Path):
        self.args = args
        self.save_dir = save_dir

    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        pass

    def observe_subgoal(
        self,
        subgoal: Optional[str],
        observation_before_subgoal: np.ndarray,
        step_idx: int,
        previous_subgoal_completed: Optional[bool] = None,
    ) -> None:
        pass

    def step(
        self,
        epstate: EpisodeState,
        subgoal: Optional[str],
    ) -> Optional[bool]:
        return None

    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        pass


class NullReporter(ReporterBase):
    """No-op Reporter used until a concrete Reporter is configured."""


class QwenVLReporter(ReporterBase):
    """Use the original, non-fine-tuned Qwen3-VL model as Reporter."""

    def __init__(self, args, save_dir: Path):
        super().__init__(args, save_dir)
        print(f"Loading Reporter model from {args.reporter_model_path}")
        self.engine = PtEngine(
            model_id_or_path=args.reporter_model_path,
            attn_impl="sdpa",
        )
        self.current_subgoal: Optional[str] = None
        self.observation_before_path: Optional[Path] = None
        self.frames_dir: Optional[Path] = None
        self.init_frames_dir: Optional[Path] = None
        self.log_path: Optional[Path] = None

    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        self.current_subgoal = None
        self.observation_before_path = None
        self.frames_dir = (
            self.save_dir
            / env_runner.env_id
            / "frames"
            / f"ep{env_runner.episode_id}"
        )
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.init_frames_dir = (
            self.save_dir
            / env_runner.env_id
            / "init_frames"
            / f"ep{env_runner.episode_id}"
        )
        if self.init_frames_dir.exists():
            shutil.rmtree(self.init_frames_dir)
        self.init_frames_dir.mkdir(parents=True, exist_ok=True)
        log_dir = self.save_dir / env_runner.env_id / "reporter_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / (
            f"{env_runner.env_id}_ep{env_runner.episode_id}.log"
        )
        self.log_path.write_text("", encoding="utf-8")

    def observe_subgoal(
        self,
        subgoal: Optional[str],
        observation_before_subgoal: np.ndarray,
        step_idx: int,
        previous_subgoal_completed: Optional[bool] = None,
    ) -> None:
        if subgoal is None:
            return
        if (
            subgoal == self.current_subgoal
            and previous_subgoal_completed is not True
        ):
            return
        if self.frames_dir is None or self.init_frames_dir is None:
            return

        self.current_subgoal = subgoal
        frame_path = self.frames_dir / f"step_{step_idx}_image.png"
        if not frame_path.exists():
            imageio.imwrite(frame_path, observation_before_subgoal)

        self.observation_before_path = (
            self.init_frames_dir / f"step_{step_idx}_image.png"
        )
        if not self.observation_before_path.exists():
            imageio.imwrite(
                self.observation_before_path,
                observation_before_subgoal,
            )

    def step(
        self,
        epstate: EpisodeState,
        subgoal: Optional[str],
    ) -> Optional[bool]:
        if (
            subgoal is None
            or self.observation_before_path is None
            or self.frames_dir is None
            or self.log_path is None
        ):
            return None

        current_observation, _, _ = epstate.get_current_obs()
        step_idx = epstate.count
        current_path = self.frames_dir / f"step_{step_idx}_image.png"
        if not current_path.exists():
            imageio.imwrite(current_path, current_observation)

        request = {
            # The order matches the two <image> placeholders in the user prompt.
            "images": [str(self.observation_before_path), str(current_path)],
            "messages": [
                {"role": "system", "content": REPORTER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": REPORTER_USER_PROMPT.replace("{subgoal}", subgoal),
                },
            ],
        }
        response = self.engine.infer(
            [InferRequest(**request)],
            request_config=RequestConfig(max_tokens=64, temperature=0),
        )[0].choices[0].message.content
        reporter_success = self._parse_success(response)

        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                f"\nStep: {step_idx}\n"
                f"{pprint.pformat(request, width=100, sort_dicts=False)}\n"
                f"Response: {response}\n"
                f"Parsed success: {reporter_success}\n"
            )

        print(f"[robomme] Reporter response: {response}")
        return reporter_success

    @staticmethod
    def _parse_success(response: str) -> Optional[bool]:
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


def build_reporter(args, save_dir: Path) -> ReporterBase:
    if args.reporter_type == "none":
        return NullReporter(args, save_dir)
    if args.reporter_type == "qwenvl":
        return QwenVLReporter(args, save_dir)
    raise ValueError(f"Unsupported Reporter type: {args.reporter_type}")
