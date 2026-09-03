"""Reporter implementations for RoboMME evaluation."""

import pprint
import shutil
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
from swift.llm import InferRequest, PtEngine, RequestConfig

from env_runner import EnvRunner
from mme_vla_suite.reporter_evaluation import debounce_reporter_success
from mme_vla_suite.reporter_evaluation import parse_reporter_success
from mme_vla_suite.reporter_prompts import (
    REPORTER_SYSTEM_PROMPT,
    format_reporter_user_prompt,
)
from utils import EpisodeState


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
    """Use Qwen3-VL, optionally with a fine-tuned Reporter adapter."""

    def __init__(self, args, save_dir: Path):
        super().__init__(args, save_dir)
        adapter_path = getattr(args, "reporter_adapter_path", "")
        print(
            f"Loading Reporter model from {args.reporter_model_path}"
            + (f" with adapter {adapter_path}" if adapter_path else "")
        )
        engine_kwargs = dict(
            model_id_or_path=args.reporter_model_path,
            attn_impl="sdpa",
        )
        if adapter_path:
            engine_kwargs["adapters"] = [adapter_path]
        self.engine = PtEngine(**engine_kwargs)
        self.current_subgoal: Optional[str] = None
        self.observation_before_path: Optional[Path] = None
        self.frames_dir: Optional[Path] = None
        self.init_frames_dir: Optional[Path] = None
        self.log_path: Optional[Path] = None
        self.reporter_debounce = bool(getattr(args, "reporter_debounce", True))
        self.consecutive_true_count = 0

    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        self.current_subgoal = None
        self.observation_before_path = None
        self.consecutive_true_count = 0
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
            or self.init_frames_dir is None
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
                    "content": format_reporter_user_prompt(subgoal),
                },
            ],
        }
        response = self.engine.infer(
            [InferRequest(**request)],
            request_config=RequestConfig(max_tokens=64, temperature=0),
        )[0].choices[0].message.content
        raw_reporter_success = self._parse_success(response)
        if self.reporter_debounce:
            reporter_success = debounce_reporter_success(
                raw_reporter_success,
                self.consecutive_true_count,
            )
            self.consecutive_true_count = (
                self.consecutive_true_count + 1
                if raw_reporter_success is True
                else 0
            )
        else:
            # dev_trinity behavior: no filtering between Reporter and Manager.
            reporter_success = raw_reporter_success
            self.consecutive_true_count = 0

        next_init_path = None
        if reporter_success is True:
            # The completed subgoal ends at the current observation. Promote
            # that exact frame before returning so the next Manager prediction
            # and every later Reporter comparison use the newest init frame.
            next_init_path = self.init_frames_dir / f"step_{step_idx}_image.png"
            if not next_init_path.exists():
                shutil.copy2(current_path, next_init_path)
            self.observation_before_path = next_init_path

        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                f"\nStep: {step_idx}\n"
                f"{pprint.pformat(request, width=100, sort_dicts=False)}\n"
                f"Response: {response}\n"
                f"Parsed success: {raw_reporter_success}\n"
                f"Debounce enabled: {self.reporter_debounce}\n"
                f"Effective success: {reporter_success}\n"
            )
            if next_init_path is not None:
                log_file.write(f"Next init frame: {next_init_path}\n")

        print(f"[robomme] Reporter response: {response}")
        return reporter_success

    @staticmethod
    def _parse_success(response: str) -> Optional[bool]:
        return parse_reporter_success(response)


def build_reporter(args, save_dir: Path) -> ReporterBase:
    if args.reporter_type == "none":
        return NullReporter(args, save_dir)
    if args.reporter_type == "qwenvl":
        return QwenVLReporter(args, save_dir)
    raise ValueError(f"Unsupported Reporter type: {args.reporter_type}")
