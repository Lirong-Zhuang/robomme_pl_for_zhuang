"""Build QwenVL Reporter completion-classification datasets.

The frame selection and positive-sample duplication rules intentionally reuse
the Manager QwenVL builder.  Each row contains the observation where a
subgoal began followed by a capacity-limited sliding window of observations from
Reporter calls in that subgoal. The window is not padded while it fills.
"""

from __future__ import annotations

import json
import os
from collections import deque
from collections.abc import Collection, Mapping
from typing import Literal

import cv2
import h5py
import imageio.v2 as imageio

from mme_vla_suite.dataset_builder.build_manager_dataset_qwenvl import (
    DatasetBuilder as ManagerDatasetBuilder,
)
from mme_vla_suite.dataset_builder.robomme_h5_utils import (
    get_task_goal,
    get_timestep_indices,
    resolve_subgoal,
)
from mme_vla_suite.reporter_prompts import (
    DEFAULT_REPORTER_HISTORY_SIZE,
    REPORTER_SYSTEM_PROMPT,
    build_reporter_image_window,
    format_reporter_user_prompt,
    validate_reporter_history_size,
)


class DatasetBuilder(ManagerDatasetBuilder):
    """Create simple and grounded Reporter JSONL data from RoboMME HDF5."""

    def __init__(
        self,
        raw_data_path: str = "data/robomme_h5_data",
        preprocessed_data_path: str = "data/robomme_preprocessed_data",
        max_episodes: int | None = None,
        visualize: bool = False,
        reporter_dir_name: str = "reporter_qwenvl",
        task_names: list[str] | None = None,
        episode_indices_by_task: Mapping[str, Collection[int]] | None = None,
        duplicate_samples: bool = True,
        data_split: Literal["train", "test"] = "train",
        reporter_history_size: int = DEFAULT_REPORTER_HISTORY_SIZE,
    ) -> None:
        reporter_history_size = validate_reporter_history_size(
            reporter_history_size
        )
        super().__init__(
            raw_data_path=raw_data_path,
            preprocessed_data_path=preprocessed_data_path,
            max_episodes=max_episodes,
            visualize=visualize,
            manager_dir_name=reporter_dir_name,
            task_names=task_names,
            episode_indices_by_task=episode_indices_by_task,
            duplicate_samples=duplicate_samples,
            data_split=data_split,
        )
        self.reporter_history_size = reporter_history_size

    @staticmethod
    def make_reporter_data(
        subgoal: str,
        image_paths: list[str],
        success: bool,
        history_size: int = DEFAULT_REPORTER_HISTORY_SIZE,
    ) -> dict:
        """Format one row exactly like the live Reporter request and response."""
        expected_image_count = validate_reporter_history_size(history_size) + 1
        if len(image_paths) != expected_image_count:
            raise ValueError(
                f"Reporter row requires {expected_image_count} images "
                f"(init + {history_size} observations), got {len(image_paths)}"
            )
        return {
            "messages": [
                {"role": "system", "content": REPORTER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": format_reporter_user_prompt(subgoal, history_size),
                },
                {
                    "role": "assistant",
                    "content": json.dumps({"success": success}),
                },
            ],
            "images": image_paths,
        }

    def _append_reporter_rows(
        self,
        simple_data: dict,
        grounded_data: dict,
        times: int = 1,
    ) -> None:
        """Append both subgoal variants, including requested duplicates."""
        for _ in range(times):
            with open(
                self.simple_subgoal_data_path,
                "a",
                encoding="utf-8",
            ) as file:
                file.write(json.dumps(simple_data) + "\n")
            with open(
                self.grounded_subgoal_data_path,
                "a",
                encoding="utf-8",
            ) as file:
                file.write(json.dumps(grounded_data) + "\n")

    def _write_frame(
        self,
        episode_data: h5py.Group,
        env_id: str,
        episode_idx: int,
        step_idx: int,
    ) -> str:
        image_path = os.path.join(
            self.images_dir,
            f"{env_id}_ep{episode_idx}_step{step_idx}.png",
        )
        if not os.path.exists(image_path):
            image = episode_data[f"timestep_{step_idx}"]["obs"]["front_rgb"][()]
            imageio.imwrite(image_path, image)
        return image_path

    @staticmethod
    def _subgoals_at_step(
        episode_data: h5py.Group,
        step_idx: int,
        last_simple_subgoal: str | None = None,
        last_grounded_subgoal: str | None = None,
    ) -> tuple[str, str]:
        info = episode_data[f"timestep_{step_idx}"]["info"]
        simple = info["simple_subgoal"][()].decode().lower()
        grounded = info["grounded_subgoal"][()].decode().lower()
        return (
            resolve_subgoal(simple, last_simple_subgoal),
            resolve_subgoal(grounded, last_grounded_subgoal),
        )

    def process_per_episode(
        self,
        env_dataset: h5py.File,
        env_id: str,
        episode_idx: int,
    ) -> dict[str, int]:
        """Build Reporter rows for one episode."""
        print(f"processing Reporter episode {episode_idx} of {env_id}...")
        episode_data = env_dataset[f"episode_{episode_idx}"]
        # Access the goal as a schema validation shared with the Manager builder.
        get_task_goal(episode_data, lower=True)
        timestep_indices = get_timestep_indices(episode_data)
        num_timesteps = len(timestep_indices)
        exec_start_idx = self._first_execution_step(episode_data)

        transition_idxs = self._compute_transition_idxs(
            episode_data,
            env_id,
            exec_start_idx,
            timestep_indices,
        )
        if transition_idxs[-1] != num_timesteps - 1:
            transition_idxs.append(num_timesteps - 1)

        select_idxs, duplicate_idxs = self._compute_select_and_duplicate_idxs(
            transition_idxs,
            num_timesteps,
            env_id,
        )
        selected = set(idx for idx in select_idxs if idx >= exec_start_idx)
        print("transition_idxs: ", transition_idxs)
        print("select_idxs: ", sorted(selected))

        positive_rows = 0
        negative_rows = 0
        duplicate_rows = 0
        last_simple_subgoal = None
        last_grounded_subgoal = None
        visualization_frames = []
        for start_idx, end_idx in zip(transition_idxs[:-1], transition_idxs[1:]):
            simple_subgoal, grounded_subgoal = self._subgoals_at_step(
                episode_data,
                start_idx,
                last_simple_subgoal,
                last_grounded_subgoal,
            )
            last_simple_subgoal = simple_subgoal
            last_grounded_subgoal = grounded_subgoal
            before_path = self._write_frame(
                episode_data,
                env_id,
                episode_idx,
                start_idx,
            )
            reporter_observation_paths: deque[str] = deque(
                maxlen=self.reporter_history_size
            )
            # Reporter is called after execution chunks, never on the initial
            # frame. RoboMME terminates at the final frame before another call.
            current_idxs = sorted(
                idx
                for idx in selected
                if start_idx < idx <= end_idx and idx != num_timesteps - 1
            )
            for idx in current_idxs:
                success = idx == end_idx
                after_path = self._write_frame(
                    episode_data,
                    env_id,
                    episode_idx,
                    idx,
                )
                # ``selected`` mirrors online Reporter invocation points. The
                # queue starts empty for every subgoal and is never padded.
                reporter_observation_paths.append(after_path)
                image_paths = build_reporter_image_window(
                    before_path,
                    reporter_observation_paths,
                    self.reporter_history_size,
                )
                simple_data = self.make_reporter_data(
                    simple_subgoal,
                    image_paths,
                    success,
                    len(reporter_observation_paths),
                )
                grounded_data = self.make_reporter_data(
                    grounded_subgoal,
                    image_paths,
                    success,
                    len(reporter_observation_paths),
                )
                self._append_reporter_rows(simple_data, grounded_data)

                if success:
                    positive_rows += 1
                else:
                    negative_rows += 1

                dup_count = (
                    duplicate_idxs.get(idx, 0)
                    if self.duplicate_samples
                    else 0
                )
                if dup_count:
                    print(f"duplicate Reporter step {idx} for {dup_count} more times")
                    self._append_reporter_rows(
                        simple_data,
                        grounded_data,
                        times=dup_count,
                    )
                    duplicate_rows += dup_count

                if self.visualize:
                    combined = cv2.hconcat(
                        [cv2.imread(image_path) for image_path in image_paths]
                    )
                    label = "true" if success else "false"
                    cv2.putText(
                        combined,
                        (
                            f"Step {start_idx}->{idx}; "
                            f"window={len(reporter_observation_paths)}/"
                            f"{self.reporter_history_size}; success={label}"
                        ),
                        (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    visualization_frames.extend([combined] * (dup_count + 1))

        if self.visualize and visualization_frames:
            visualization_dir = os.path.join(self.data_dir, "visualization")
            os.makedirs(visualization_dir, exist_ok=True)
            imageio.mimsave(
                os.path.join(
                    visualization_dir,
                    f"{env_id}_ep{episode_idx}_reporter.mp4",
                ),
                visualization_frames,
                fps=1,
            )

        counts = {
            "positive": positive_rows,
            "negative": negative_rows,
            "duplicates": duplicate_rows,
        }
        print("Reporter rows: ", counts)
        return counts
