"""Base class for Manager dataset builders (QwenVL, MemER).

Shared setup, H5 iteration, and grounded-subgoal/bbox helpers.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Literal

import cv2
import h5py
import numpy as np

from mme_vla_suite.dataset_builder.robomme_h5_utils import (
    add_noise_to_bbox,
    first_execution_step,
    get_env_id_from_filename,
    get_episode_indices,
    preprocess_grounded_subgoal,
    wrap_history_subgoals,
)

if TYPE_CHECKING:
    pass


class BaseManagerDatasetBuilder:
    """Base for building Manager JSONL datasets from RoboMME HDF5."""

    def __init__(
        self,
        raw_data_path: str = "data/robomme_h5_data",
        preprocessed_data_path: str = "data/robomme_preprocessed_data",
        max_episodes: int | None = None,
        visualize: bool = False,
        manager_dir_name: str = "vlm_subgoal",
        task_names: list[str] | None = None,
        episode_indices_by_task: Mapping[str, Collection[int]] | None = None,
        duplicate_samples: bool = True,
        data_split: Literal["train", "test"] = "train",
    ) -> None:
        self.raw_data_path = raw_data_path
        self.preprocessed_data_path = preprocessed_data_path
        self.max_episodes = max_episodes
        self.visualize = visualize
        self.task_names = set(task_names) if task_names else None
        self.duplicate_samples = duplicate_samples
        self.data_split = data_split
        self.episode_indices_by_task = (
            {
                task_name: set(episode_indices)
                for task_name, episode_indices in episode_indices_by_task.items()
            }
            if episode_indices_by_task is not None
            else None
        )

        available_tasks = {
            get_env_id_from_filename(file)
            for file in os.listdir(self.raw_data_path)
            if file.endswith(".h5")
        }
        if not available_tasks:
            raise ValueError(
                f"No .h5 files found directly under {self.raw_data_path!r}"
            )
        if self.task_names is not None:
            unknown_tasks = self.task_names - available_tasks
            if unknown_tasks:
                raise ValueError(
                    f"Unknown task(s): {sorted(unknown_tasks)}. "
                    f"Available tasks: {sorted(available_tasks)}"
                )

        self.data_dir = os.path.join(preprocessed_data_path, manager_dir_name)
        self.images_dir = os.path.join(self.data_dir, "images")
        self.simple_subgoal_data_path = os.path.join(
            self.data_dir, f"simple_subgoal_{data_split}.jsonl"
        )
        self.grounded_subgoal_data_path = os.path.join(
            self.data_dir, f"grounded_subgoal_{data_split}.jsonl"
        )
        self._setup_output_dirs()
        self.history_simple_subgoals = []
        self.history_grounded_subgoals = []
        self.history_grounded_bboxes = []

    def _setup_output_dirs(self) -> None:
        if os.path.exists(self.images_dir):
            shutil.rmtree(self.images_dir)
        os.makedirs(self.images_dir, exist_ok=True)
        if os.path.exists(self.simple_subgoal_data_path):
            os.remove(self.simple_subgoal_data_path)
        if os.path.exists(self.grounded_subgoal_data_path):
            os.remove(self.grounded_subgoal_data_path)

    def run(self) -> list:
        """Process all H5 files and episodes. Returns list of process_per_episode return values."""
        results: list = []
        for file in os.listdir(self.raw_data_path):
            if not file.endswith(".h5"):
                continue
            if (
                self.task_names is not None
                and get_env_id_from_filename(file) not in self.task_names
            ):
                continue
            print(f"\nprocessing file: {file}")
            with h5py.File(os.path.join(self.raw_data_path, file), "r") as data:
                env_id = get_env_id_from_filename(file)
                episode_indices = get_episode_indices(data, self.max_episodes)
                if self.episode_indices_by_task is not None:
                    selected = self.episode_indices_by_task.get(env_id, set())
                    episode_indices = [
                        episode_idx
                        for episode_idx in episode_indices
                        if episode_idx in selected
                    ]
                for episode_idx in episode_indices:
                    r = self.process_per_episode(data, env_id, episode_idx)
                    results.append(r)
        return results

    def process_per_episode(
        self,
        env_dataset: h5py.File,
        env_id: str,
        episode_idx: int,
    ):
        """Process one episode; subclasses must implement. Return value is builder-specific."""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Shared helpers (delegate to robomme_h5_utils)
    # -------------------------------------------------------------------------

    def _first_execution_step(self, episode_data: h5py.Group) -> int:
        return first_execution_step(episode_data)

    def _preprocess_grounded_subgoal(self, subgoal: str) -> tuple[str, list]:
        return preprocess_grounded_subgoal(subgoal)

    def _add_noise_to_bbox(self, bbox: list) -> list:
        return add_noise_to_bbox(bbox)

    def _wrap_history_subgoals(self, subgoals: list) -> str:
        return wrap_history_subgoals(subgoals)

    def combine_image_and_wrist_image(
        self,
        image: np.ndarray,
        wrist_image: np.ndarray,
        simple_subgoal: str,
    ) -> np.ndarray:
        """Horizontal stack of image and wrist_image with subgoal text overlay."""
        output = np.concatenate([image, wrist_image], axis=1)
        output = cv2.putText(
            output,
            simple_subgoal,
            (10, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        return output
