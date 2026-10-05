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
import numpy as np

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
    REPORTER_PROMPT_VERSION,
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
        self.failure_hard_negative_records: list[dict] = []

    def run(self) -> list:
        """Build rows and write a machine-readable failed-grasp label audit."""
        results = super().run()
        audit_path = os.path.join(
            self.data_dir,
            "failure_hard_negative_audit.json",
        )
        audit = {
            "reporter_prompt_version": REPORTER_PROMPT_VERSION,
            "data_split": self.data_split,
            "failure_recovery_episode_count": len(
                self.failure_hard_negative_records
            ),
            "failure_hard_negative_count": len(
                self.failure_hard_negative_records
            ),
            "all_labels_are_false": all(
                record["success"] is False
                for record in self.failure_hard_negative_records
            ),
            "records": self.failure_hard_negative_records,
        }
        with open(audit_path, "w", encoding="utf-8") as file:
            json.dump(audit, file, indent=2, ensure_ascii=False)
            file.write("\n")
        print(
            "Failure hard-negative audit: "
            f"{len(self.failure_hard_negative_records)} verified rows -> "
            f"{audit_path}"
        )
        return results

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

    @staticmethod
    def _attribute_text(group: h5py.Group, name: str) -> str | None:
        value = group.attrs.get(name)
        if isinstance(value, np.ndarray):
            value = value.reshape(-1)[0] if value.size else None
        if isinstance(value, (bytes, np.bytes_)):
            value = value.decode("utf-8")
        return None if value is None else str(value)

    @classmethod
    def _is_failure_recovery_episode(cls, episode_data: h5py.Group) -> bool:
        origin = cls._attribute_text(episode_data, "dataset_origin")
        return origin == "failure_recovery" or bool(
            episode_data.attrs.get("failure_recovery", False)
        )

    @staticmethod
    def _waypoint_at_step(
        episode_data: h5py.Group,
        step_idx: int,
    ) -> np.ndarray | None:
        path = f"timestep_{step_idx}/action/waypoint_action"
        if path not in episode_data:
            return None
        try:
            waypoint = np.asarray(episode_data[path][()]).reshape(-1)
        except (TypeError, ValueError):
            return None
        if waypoint.shape != (7,) or not np.all(np.isfinite(waypoint)):
            return None
        return waypoint

    @classmethod
    def _failed_grasp_endpoint_in_span(
        cls,
        episode_data: h5py.Group,
        start_idx: int,
        end_idx: int,
    ) -> int | None:
        """Locate the end of the injected failed-grasp attempt.

        RoboMME records each planner waypoint as
        ``[position(3), rpy(3), gripper]``.  The injected attempt is the unique
        ``open -> close -> open`` sequence that returns to the same ready pose,
        followed by a later close waypoint for the successful retry.  Selecting
        the final frame of the return/open run gives the Reporter the completed
        *motion* while the object is still not grasped.
        """
        runs: list[dict] = []
        for step_idx in range(start_idx, end_idx):
            waypoint = cls._waypoint_at_step(episode_data, step_idx)
            if waypoint is None:
                continue
            if runs and np.array_equal(waypoint, runs[-1]["waypoint"]):
                runs[-1]["end"] = step_idx
            else:
                runs.append(
                    {
                        "start": step_idx,
                        "end": step_idx,
                        "waypoint": waypoint,
                    }
                )

        for run_idx in range(len(runs) - 2):
            first, second, third = runs[run_idx : run_idx + 3]
            grippers = tuple(
                1 if run["waypoint"][-1] > 0 else -1
                for run in (first, second, third)
            )
            returns_to_ready_pose = np.allclose(
                first["waypoint"][:6],
                third["waypoint"][:6],
                rtol=0,
                atol=1e-5,
            )
            has_later_close = any(
                run["waypoint"][-1] < 0
                for run in runs[run_idx + 3 :]
            )
            if (
                grippers == (1, -1, 1)
                and returns_to_ready_pose
                and has_later_close
            ):
                return int(third["end"])
        return None

    @classmethod
    def _failure_hard_negative_idxs(
        cls,
        episode_data: h5py.Group,
        transition_idxs: list[int],
    ) -> list[int]:
        candidates = []
        for start_idx, end_idx in zip(transition_idxs[:-1], transition_idxs[1:]):
            candidate = cls._failed_grasp_endpoint_in_span(
                episode_data,
                start_idx,
                end_idx,
            )
            if candidate is not None:
                candidates.append(candidate)
        return sorted(set(candidates))

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

        failure_hard_negative_idxs: list[int] = []
        is_failure_recovery = self._is_failure_recovery_episode(episode_data)
        if is_failure_recovery:
            failure_hard_negative_idxs = self._failure_hard_negative_idxs(
                episode_data,
                transition_idxs,
            )
            if len(failure_hard_negative_idxs) != 1:
                failure_mode = self._attribute_text(
                    episode_data,
                    "failure_mode",
                )
                raise RuntimeError(
                    f"{env_id} episode_{episode_idx} is marked as a failure-"
                    "recovery episode, but the builder found "
                    f"{len(failure_hard_negative_idxs)} injected failed-grasp "
                    "endpoints (expected exactly 1). "
                    f"failure_mode={failure_mode!r}. Refusing to build an "
                    "unverified Reporter label."
                )

        # Semantic transitions and failure anchors intentionally have different
        # meanings. Both split the approximately stride-sized sampling ranges,
        # but only a semantic transition completes a subgoal, resets Reporter
        # history, and may be duplicated as a positive training sample.
        sampling_anchor_idxs = sorted(
            set(transition_idxs + failure_hard_negative_idxs)
        )
        select_idxs, _ = self._compute_select_and_duplicate_idxs(
            sampling_anchor_idxs,
            num_timesteps,
            env_id,
        )
        _, duplicate_idxs = self._compute_select_and_duplicate_idxs(
            transition_idxs,
            num_timesteps,
            env_id,
        )
        selected = set(idx for idx in select_idxs if idx >= exec_start_idx)
        missing_failure_anchors = set(failure_hard_negative_idxs) - selected
        if missing_failure_anchors:
            raise RuntimeError(
                f"{env_id} episode_{episode_idx}: failure sampling anchors "
                f"were not selected: {sorted(missing_failure_anchors)}"
            )
        print("transition_idxs: ", transition_idxs)
        print("sampling_anchor_idxs: ", sampling_anchor_idxs)
        print("select_idxs: ", sorted(selected))
        if failure_hard_negative_idxs:
            print(
                "failure_hard_negative_idxs: ",
                failure_hard_negative_idxs,
                "-> success=false",
            )

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
                if idx in failure_hard_negative_idxs and success:
                    raise RuntimeError(
                        f"{env_id} episode_{episode_idx} step {idx}: failed "
                        "grasp endpoint overlaps a completed-subgoal label"
                    )
                after_path = self._write_frame(
                    episode_data,
                    env_id,
                    episode_idx,
                    idx,
                )
                # Selected observations follow the regular sampling schedule
                # plus a failure anchor when present. The queue resets only at
                # real subgoal transitions and is never padded.
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

                if idx in failure_hard_negative_idxs:
                    self.failure_hard_negative_records.append(
                        {
                            "task": env_id,
                            "episode": episode_idx,
                            "step": idx,
                            "failure_event_step": idx,
                            "reporter_sample_step": idx,
                            "success": False,
                            "simple_subgoal": simple_subgoal,
                            "failure_mode": self._attribute_text(
                                episode_data,
                                "failure_mode",
                            ),
                            "difficulty": self._attribute_text(
                                episode_data,
                                "difficulty",
                            ),
                        }
                    )

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
