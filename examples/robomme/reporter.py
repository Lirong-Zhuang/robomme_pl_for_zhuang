"""Reporter implementations for RoboMME evaluation."""

import os
import pprint
import shutil
from collections import deque
from pathlib import Path
from typing import Optional

import imageio
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from swift.llm import InferRequest, PtEngine, RequestConfig

from env_runner import EnvRunner
from mme_vla_suite.reporter_gripper_tool import PickUpGripperGate
from mme_vla_suite.reporter_evaluation import debounce_reporter_success
from mme_vla_suite.reporter_evaluation import parse_reporter_success
from mme_vla_suite.prompts import DEFAULT_REPORTER_PROMPT_VERSION
from mme_vla_suite.prompts import get_reporter_prompt
from mme_vla_suite.reporter_prompts import (
    DEFAULT_REPORTER_HISTORY_SIZE,
    build_reporter_multiview_window,
    validate_reporter_history_size,
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
        overall_observation: np.ndarray,
        wrist_observation: np.ndarray,
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
        self.reporter_prompt = get_reporter_prompt(
            getattr(
                args,
                "reporter_prompt_version",
                DEFAULT_REPORTER_PROMPT_VERSION,
            )
        )
        if (
            not self.reporter_prompt.include_init_image
            or self.reporter_prompt.init_image_views != ("overall", "wrist")
            or self.reporter_prompt.image_views != ("overall", "wrist")
        ):
            raise ValueError(
                "Online Reporter evaluation requires the dual-view prompt "
                "dual_view_temporal_v3 (overall + wrist for init and history); "
                f"got {self.reporter_prompt.version!r}"
            )
        print(
            f"Loading Reporter {args.reporter_qwen_backbone_name} "
            f"from {args.reporter_qwen_backbone_path}"
            + (f" with adapter {adapter_path}" if adapter_path else "")
        )
        engine_kwargs = dict(
            model_id_or_path=args.reporter_qwen_backbone_path,
            attn_impl="sdpa",
        )
        if adapter_path:
            engine_kwargs["adapters"] = [adapter_path]
        self.engine = PtEngine(**engine_kwargs)
        self.current_subgoal: Optional[str] = None
        self.observation_before_paths: Optional[tuple[Path, Path]] = None
        self.overall_frames_dir: Optional[Path] = None
        self.wrist_frames_dir: Optional[Path] = None
        self.init_episode_dir: Optional[Path] = None
        self.init_overall_dir: Optional[Path] = None
        self.init_wrist_dir: Optional[Path] = None
        self.log_path: Optional[Path] = None
        self.reporter_debounce = bool(getattr(args, "reporter_debounce", True))
        self.reporter_history_size = validate_reporter_history_size(
            int(
                getattr(
                    args,
                    "reporter_history_size",
                    DEFAULT_REPORTER_HISTORY_SIZE,
                )
            )
        )
        self.observation_history: deque[tuple[Path, Path]] = deque(
            maxlen=self.reporter_history_size
        )
        self.consecutive_true_count = 0
        self.gripper_gate = (
            PickUpGripperGate(
                closed_position=float(args.reporter_gripper_closed_position),
                open_position=float(args.reporter_gripper_open_position),
                endpoint_buffer=float(args.reporter_gripper_endpoint_buffer),
            )
            if bool(getattr(args, "reporter_gripper_gate", False))
            else None
        )

    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        self.current_subgoal = None
        self.observation_before_paths = None
        self.observation_history.clear()
        self.consecutive_true_count = 0
        frames_root = self.save_dir / env_runner.env_id / "frames"
        self.overall_frames_dir = (
            frames_root / "overall" / f"ep{env_runner.episode_id}"
        )
        self.wrist_frames_dir = (
            frames_root / "wrist" / f"ep{env_runner.episode_id}"
        )
        self.overall_frames_dir.mkdir(parents=True, exist_ok=True)
        self.wrist_frames_dir.mkdir(parents=True, exist_ok=True)
        self.init_episode_dir = (
            self.save_dir
            / env_runner.env_id
            / "init_frames"
            / f"ep{env_runner.episode_id}"
        )
        if self.init_episode_dir.exists():
            shutil.rmtree(self.init_episode_dir)
        self.init_overall_dir = self.init_episode_dir / "overall"
        self.init_wrist_dir = self.init_episode_dir / "wrist"
        self.init_overall_dir.mkdir(parents=True, exist_ok=True)
        self.init_wrist_dir.mkdir(parents=True, exist_ok=True)
        log_dir = self.save_dir / env_runner.env_id / "reporter_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / (
            f"{env_runner.env_id}_ep{env_runner.episode_id}.log"
        )
        self.log_path.write_text("", encoding="utf-8")

    def _write_frame_pair(
        self,
        overall_observation: np.ndarray,
        wrist_observation: np.ndarray,
        step_idx: int,
    ) -> tuple[Path, Path]:
        if self.overall_frames_dir is None or self.wrist_frames_dir is None:
            raise RuntimeError("Reporter frame directories are not initialized")
        overall_path = self.overall_frames_dir / f"step_{step_idx}.png"
        wrist_path = self.wrist_frames_dir / f"step_{step_idx}.png"
        if not overall_path.exists():
            imageio.imwrite(overall_path, overall_observation)
        if not wrist_path.exists():
            imageio.imwrite(wrist_path, wrist_observation)
        return overall_path, wrist_path

    def _write_init_pair(
        self,
        overall_observation: np.ndarray,
        wrist_observation: np.ndarray,
        step_idx: int,
    ) -> tuple[Path, Path]:
        if self.init_overall_dir is None or self.init_wrist_dir is None:
            raise RuntimeError("Reporter init-frame directories are not initialized")
        overall_path = self.init_overall_dir / f"step_{step_idx}.png"
        wrist_path = self.init_wrist_dir / f"step_{step_idx}.png"
        imageio.imwrite(overall_path, overall_observation)
        imageio.imwrite(wrist_path, wrist_observation)
        return overall_path, wrist_path

    def observe_subgoal(
        self,
        subgoal: Optional[str],
        overall_observation: np.ndarray,
        wrist_observation: np.ndarray,
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
        if self.overall_frames_dir is None or self.wrist_frames_dir is None:
            return

        self.current_subgoal = subgoal
        # Keep ordinary per-view frames for the evaluation record. The separate
        # init pair is temporary and is removed when the episode ends.
        self._write_frame_pair(
            overall_observation,
            wrist_observation,
            step_idx,
        )
        self.observation_before_paths = self._write_init_pair(
            overall_observation,
            wrist_observation,
            step_idx,
        )
        self.observation_history.clear()

    def step(
        self,
        epstate: EpisodeState,
        subgoal: Optional[str],
    ) -> Optional[bool]:
        if (
            subgoal is None
            or self.observation_before_paths is None
            or self.overall_frames_dir is None
            or self.wrist_frames_dir is None
            or self.log_path is None
        ):
            return None

        current_observation, current_wrist_observation, current_robot_state = (
            epstate.get_current_obs()
        )
        step_idx = epstate.count
        current_paths = self._write_frame_pair(
            current_observation,
            current_wrist_observation,
            step_idx,
        )
        # This is a Reporter-call history. Environment observations between
        # two calls are intentionally not inserted into the window.
        self.observation_history.append(current_paths)
        image_paths = [
            *self.observation_before_paths,
            *build_reporter_multiview_window(
                self.observation_history,
                self.reporter_history_size,
            ),
        ]

        request = {
            # Input order: init overall/wrist, then up to three recent
            # observation pairs (overall then wrist), oldest to newest.
            "images": [str(image_path) for image_path in image_paths],
            "messages": self.reporter_prompt.build_messages(
                subgoal,
                len(self.observation_history),
            ),
        }
        self.reporter_prompt.validate_media_alignment(
            request["messages"],
            image_count=len(request["images"]),
        )
        response = self.engine.infer(
            [InferRequest(**request)],
            request_config=RequestConfig(max_tokens=64, temperature=0),
        )[0].choices[0].message.content
        raw_reporter_success = self._parse_success(response)

        # The gripper tool is a strict veto and is called only for a parsed
        # Reporter true. Non-pick-up subgoals pass through as not applicable.
        # RoboMME's packed robot state stores the first observed finger opening
        # as its final value; the policy-facing state shape remains unchanged.
        gripper_gate_result = None
        verified_reporter_success = raw_reporter_success
        if raw_reporter_success is True and self.gripper_gate is not None:
            state = np.asarray(current_robot_state, dtype=np.float32).reshape(-1)
            gripper_position = float(state[-1]) if state.size else float("nan")
            gripper_gate_result = self.gripper_gate.verify_reporter_true(
                subgoal=subgoal,
                gripper_position=gripper_position,
            )
            if gripper_gate_result.applicable and not gripper_gate_result.passed:
                verified_reporter_success = False

        if self.reporter_debounce:
            reporter_success = debounce_reporter_success(
                verified_reporter_success,
                self.consecutive_true_count,
            )
            self.consecutive_true_count = (
                self.consecutive_true_count + 1
                if verified_reporter_success is True
                else 0
            )
        else:
            # dev_trinity behavior: no filtering between Reporter and Manager.
            reporter_success = verified_reporter_success
            self.consecutive_true_count = 0

        if reporter_success is True:
            # The completed subgoal must not leak temporal context into the
            # next one. Clear immediately, before Manager consumes the result.
            self.observation_history.clear()

        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                f"\nStep: {step_idx}\n"
                f"{pprint.pformat(request, width=100, sort_dicts=False)}\n"
                f"Response: {response}\n"
                f"Parsed success: {raw_reporter_success}\n"
                f"Success after gripper gate: {verified_reporter_success}\n"
                f"Gripper gate enabled: {self.gripper_gate is not None}\n"
                f"Debounce enabled: {self.reporter_debounce}\n"
                f"Effective success: {reporter_success}\n"
                f"Reporter history size: {self.reporter_history_size}\n"
                f"Reporter prompt version: {self.reporter_prompt.version}\n"
                f"Reporter prompt sha256: {self.reporter_prompt.content_hash}\n"
            )
            if gripper_gate_result is not None:
                log_file.write(
                    "Gripper gate result: "
                    f"applicable={gripper_gate_result.applicable}; "
                    f"passed={gripper_gate_result.passed}; "
                    f"reason={gripper_gate_result.reason}; "
                    f"position={gripper_gate_result.gripper_position:.6f}; "
                    "accepted_interval=("
                    f"{gripper_gate_result.accepted_lower_bound:.6f}, "
                    f"{gripper_gate_result.accepted_upper_bound:.6f})\n"
                )
            log_file.write("Reporter views: overall,wrist\n")
            log_file.write(
                "Reporter init frames: "
                f"{self.observation_before_paths[0]}, "
                f"{self.observation_before_paths[1]}\n"
            )

        print(f"[robomme] Reporter response: {response}")
        return reporter_success

    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        init_root = (
            self.init_episode_dir.parent
            if self.init_episode_dir is not None
            else None
        )
        if self.init_episode_dir is not None and self.init_episode_dir.exists():
            shutil.rmtree(self.init_episode_dir)
        if init_root is not None and init_root.exists():
            try:
                init_root.rmdir()
            except OSError:
                # Another episode may still have temporary init frames.
                pass
        self.observation_before_paths = None
        self.observation_history.clear()
        self.init_episode_dir = None
        self.init_overall_dir = None
        self.init_wrist_dir = None

    @staticmethod
    def _parse_success(response: str) -> Optional[bool]:
        return parse_reporter_success(response)


def build_reporter(args, save_dir: Path) -> ReporterBase:
    if args.reporter_type == "none":
        return NullReporter(args, save_dir)
    if args.reporter_type == "qwenvl":
        return QwenVLReporter(args, save_dir)
    raise ValueError(f"Unsupported Reporter type: {args.reporter_type}")
