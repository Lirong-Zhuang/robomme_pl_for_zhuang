import os
import numpy as np
import cv2
import imageio
import re
import collections
import uuid
from pathlib import Path
from typing import Tuple, Optional

from mme_vla_suite.prompts import get_manager_prompt
from mme_vla_suite.prompts import get_reporter_prompt
from mme_vla_suite.reporter_gripper_tool import PickUpGripperGate
from mme_vla_suite.reporter_prompts import DEFAULT_REPORTER_HISTORY_SIZE



TASK_WITH_VIDEO_DEMO = [
    "VideoUnmask", "VideoUnmaskSwap", "VideoPlaceButton", "VideoPlaceOrder",
    "VideoRepick", "MoveCube", "InsertPeg", "PatternLock", "RouteStick"
]

TASK_NAME_LIST=  [      
    "BinFill",
    "StopCube",
    "PickXtimes",
    "SwingXtimes",
    
    "ButtonUnmask",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmaskSwap",
    
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick"
]

SUBGOAL_TYPES = ("simple_subgoal", "grounded_subgoal")



def pack_buffer(image_buffer, state_buffer, exec_start_idx=0):
    image_output = np.stack(image_buffer, axis=0).astype(np.uint8)[:, None]
    state_output = np.stack(state_buffer, axis=0).astype(np.float32)
    return {
        "images": image_output,
        "state": state_output,
        "add_buffer": True,
        "exec_start_idx": exec_start_idx,
    }
    
def check_args(args):
    assert args.subgoal_type in ["simple_subgoal", "grounded_subgoal", None] and args.obs_horizon == 16
    assert args.reporter_type in ["none", "qwenvl"]
    assert getattr(
        args,
        "reporter_history_size",
        DEFAULT_REPORTER_HISTORY_SIZE,
    ) >= 1
    get_manager_prompt(args.manager_prompt_version)
    get_reporter_prompt(args.reporter_prompt_version)
    if getattr(args, "reporter_gripper_gate", False):
        PickUpGripperGate(
            closed_position=args.reporter_gripper_closed_position,
            open_position=args.reporter_gripper_open_position,
            endpoint_buffer=args.reporter_gripper_endpoint_buffer,
        )
    if args.manager_use_memer:
        args.subgoal_type = "grounded_subgoal"



class EpisodeState:
    def __init__(self):
        self.image_buffer = []
        self.wrist_image_buffer = []
        self.state_buffer = []
        self.action_plan = collections.deque()
        self.count = 0
        self.exec_start_idx = 0

    def add_observation(self, img: np.ndarray, wrist_img: np.ndarray, state: np.ndarray):
        self.image_buffer.append(img.copy())
        self.wrist_image_buffer.append(wrist_img.copy())
        self.state_buffer.append(state.copy())

    def clear_buffers(self):
        self.image_buffer.clear()
        self.wrist_image_buffer.clear()
        self.state_buffer.clear()
        self.exec_start_idx = 0

    def get_current_obs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.image_buffer[-1], self.wrist_image_buffer[-1], self.state_buffer[-1]





class RolloutRecorder:
    def __init__(
        self,
        save_dir: str,
        task_goal: str,
        fps: int = 30,
        streaming: bool = False,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.total_images = []
        self.fps = fps
        self.task_goal = task_goal
        self.streaming = streaming
        self.frame_count = 0
        self._writer = None
        self._stream_error = None
        self._stream_path = (
            self.save_dir / f".rollout-{uuid.uuid4().hex}.mp4"
            if streaming
            else None
        )

    @property
    def has_frames(self) -> bool:
        return self.frame_count > 0

    @property
    def stream_error(self) -> Exception | None:
        return self._stream_error

    def _append_frame(self, frame: np.ndarray) -> None:
        if not self.streaming:
            self.total_images.append(frame)
            self.frame_count += 1
            return
        if self._stream_error is not None:
            return
        try:
            if self._writer is None:
                self._writer = imageio.get_writer(
                    str(self._stream_path),
                    fps=self.fps,
                )
            self._writer.append_data(frame)
            self.frame_count += 1
        except Exception as error:
            self._stream_error = error
            self._close_writer()
        
    def _extract_points(self, subgoal: str):
        match = re.findall(r'<(\d+), (\d+)>', subgoal)
        points = []
        for m in match:
            points.append((int(m[0]), int(m[1])))
        return points
        
    def record(
        self,
        image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
        action: np.ndarray = None,
        is_video_demo: bool = False,
        subgoal: Optional[str] = None,
        point_yx: Optional[Tuple[int, int]] = None,
    ):
        
        concat_image = np.concatenate([image, wrist_image], axis=1)
        if is_video_demo: # add a red border
            concat_image = cv2.rectangle(concat_image, (0, 0), (concat_image.shape[1], concat_image.shape[0]), (255, 0, 0), 10)
        
        frame_text = "Frame: " + str(self.frame_count)
        frame_text_area = self.add_text_area(frame_text, concat_image.shape)
        
        goal_text = "Task Goal: " + self.task_goal
        goal_text_area = self.add_text_area(goal_text, concat_image.shape)
        
        if subgoal is not None:
            points = self._extract_points(subgoal)
            displayed_subgoal = subgoal
            if point_yx is not None:
                point = (int(point_yx[0]), int(point_yx[1]))
                if point not in points:
                    points.append(point)
                    displayed_subgoal = f"{subgoal} <{point[1]}, {point[0]}>"

            subgoal_text = "Subgoal: " + displayed_subgoal
            subgoal_text_area = self.add_text_area(subgoal_text, concat_image.shape)

            # Draw only on the overall image (the left half of concat_image).
            # concat_image is newly allocated, so this cannot mutate the
            # observation used by evaluation, grounding, or action execution.
            for point_y, point_x in points:
                if 0 <= point_y < image.shape[0] and 0 <= point_x < image.shape[1]:
                    concat_image = cv2.circle(
                        concat_image,
                        (point_x, point_y),
                        5,
                        (255, 255, 0),
                        -1,
                    )
                    
            concat_image = np.concatenate([subgoal_text_area, concat_image], axis=0)
        
        state_text = "State: " + ','.join([f"{i:.4f}" for i in state])
        state_text_area = self.add_text_area(state_text, concat_image.shape)
        
        action_text = 'Action: ' + ','.join([f"{i:.4f}" for i in action]) if action is not None else "Action:None"
        action_text_area = self.add_text_area(action_text, concat_image.shape)
        # Concatenate text area on top of image
        concat_image = np.concatenate([frame_text_area, goal_text_area, action_text_area, state_text_area, concat_image], axis=0)
        self._append_frame(concat_image)
    
    def add_text_area(self, text: str, concat_image_shape: tuple):        
        # Calculate text wrapping
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        max_width = concat_image_shape[1] - 20  # Leave 10px margin on each side
        lines = []
        words = text.replace(',', ' ').split()
        current_line = words[0]
        for word in words[1:]:
            test_line = current_line + ' ' + word
            (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            
            if text_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        
        lines.append(current_line)  # Add the last line
        
        # Create text area with dynamic height
        line_height = 20
        text_area_height = max(50, len(lines) * line_height + 10)
        text_area = np.zeros((text_area_height, concat_image_shape[1], 3), dtype=np.uint8)
        
        # Draw each line
        for i, line in enumerate(lines):
            y_position = 15 + i * line_height
            text_area = cv2.putText(text_area, line, (10, y_position), font, font_scale, (255, 255, 255), thickness)
        
        return text_area
             
    def save_video(self, filename: str):
        final_path = self.save_dir / filename
        if self.streaming:
            if self._stream_error is not None:
                raise RuntimeError(
                    f"streaming video writer failed: {self._stream_error}"
                ) from self._stream_error
            if not self.has_frames:
                raise RuntimeError("cannot save an empty streaming video")
            self._close_writer(raise_errors=True)
            os.replace(self._stream_path, final_path)
            self._stream_path = None
            return
        imageio.mimsave(str(final_path), self.total_images, fps=self.fps)

    def _close_writer(self, *, raise_errors: bool = False) -> None:
        writer = self._writer
        self._writer = None
        if writer is None:
            return
        try:
            writer.close()
        except Exception:
            if raise_errors:
                raise

    def close(self) -> None:
        """Close an unfinished stream and remove its temporary output."""
        self._close_writer()
        if self._stream_path is not None:
            try:
                self._stream_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._stream_path = None
