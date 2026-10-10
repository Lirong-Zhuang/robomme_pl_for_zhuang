import json
import os
import pprint
import re
import shutil
from typing import List

import imageio
import numpy as np

os.environ['IMAGE_MAX_TOKEN_NUM'] = '256'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '64'
os.environ['FPS_MAX_FRAMES'] = '10'

from swift.llm import PtEngine, InferRequest, RequestConfig
from mme_vla_suite.prompts import DEFAULT_MANAGER_PROMPT_VERSION
from mme_vla_suite.prompts import get_manager_prompt
from mme_vla_suite.manager_response import parse_manager_response
from mme_vla_suite.manager_response import parse_verification_passed
from mme_vla_suite.manager_response import validate_interaction_pairs

class Qwen3VLModel:
    
    def __init__(self, 
        adapter_path: str,
        subgoal_type: str = "simple_subgoal", 
        prompt_version: str = DEFAULT_MANAGER_PROMPT_VERSION,
    ):
        self.model_name = "qwenvl"
        self.subgoal_type = subgoal_type
        self.image_size = (256, 256)
        
        assert subgoal_type in ["simple_subgoal", "grounded_subgoal"]
        
        self.prompt = get_manager_prompt(prompt_version)
        self.interaction_verification_enabled = (
            self.prompt.version == "qwenvl_interaction_verify_v2"
        )
        self.last_verification_passed = None
        self.system_prompt = self.prompt.system_prompt(subgoal_type)
        
        print(f"Loading Qwen3-VL-4B-Instruct model Adapter from {adapter_path}")
        self.engine = PtEngine(
            model_id_or_path='Qwen/Qwen3-VL-4B-Instruct',
            adapters=[adapter_path],
            # attn_impl='flash_attention_2' #'sdpa'
            attn_impl='sdpa'
        )
        
    def _parse_box_patterns(self, subgoal: str, replacement: str = "scaled_coords", return_bbox: bool = False):
        """
        Parse box patterns from subgoal and replace them.
        
        Args:
            subgoal: The subgoal string containing box patterns
            replacement: Either "scaled_coords" to replace with <x, y> or "bbox" to replace with <bbox>
            return_bbox: If True, also return the list of bbox coordinates
        
        Returns:
            If return_bbox is False: modified subgoal string
            If return_bbox is True: tuple of (modified subgoal string, bbox list)
        """
        matches = re.findall(r'<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>', subgoal)
        
        qwen3_vl_image_size = (1000, 1000)
        
        if len(matches) == 0:
            if return_bbox:
                return subgoal, []
            return subgoal
        
        # Extract bbox coordinates (scaled)
        bbox = [[int(float(match[0])/qwen3_vl_image_size[1]*self.image_size[1]), 
                 int(float(match[1])/qwen3_vl_image_size[0]*self.image_size[0])] for match in matches]
        
        # Replace based on replacement type
        if replacement == "scaled_coords":
            response = re.sub(
                r'<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>',
                lambda m: f'<{int(int(m.group(1)) * self.image_size[1] / qwen3_vl_image_size[1])}, {int(int(m.group(2)) * self.image_size[0] / qwen3_vl_image_size[0])}>',
                subgoal
            )
        elif replacement == "bbox":
            response = re.sub(
                r'<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>',
                '<bbox>',
                subgoal
            )
        else:
            raise ValueError(f"Invalid replacement type: {replacement}")
        
        if return_bbox:
            return response, bbox
        return response
    
    def _parse_subgoal_for_vla(self, subgoal: str) -> str:
        """Parse subgoal and replace box patterns with scaled coordinates for VLA."""
        return self._parse_box_patterns(subgoal, replacement="scaled_coords", return_bbox=False)
    
    def _parse_grounded_subgoal(self, subgoal) -> tuple:
        """Preprocess grounded subgoal by replacing box patterns with <bbox> and extracting bbox coordinates."""
        return self._parse_box_patterns(subgoal, replacement="bbox", return_bbox=True)
    
    def start_new_episode(
        self,
        save_dir: str,
        video_query: List[np.ndarray] | None,
        task_goal: str = None,
    ) -> dict:
        self.save_dir = save_dir
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        if video_query is not None and len(video_query) > 0:
            imageio.mimsave(os.path.join(self.save_dir, f"step_0_video.mp4"), video_query, fps=30)
            self.video_path = os.path.join(self.save_dir, f"step_0_video.mp4")
        else:
            self.video_path = None
        self.task_goal = task_goal
        self.conversation_history = []
        self.total_images = []
        self.subgoals = []
        self.history_simple_subgoals = []
        self.history_grounded_subgoals = []
        self.history_grounded_bboxes = []
        self.last_response = None
        self.last_verification_passed = None
     
    def _wrap_history_subgoals(self, subgoals) -> str:
        return "; ".join([f"{i+1}. {subgoal}" for i, subgoal in enumerate(subgoals)])
    
    def _parse_grounded_subgoal(self, subgoal) -> tuple:
        bbox = []
        # seatch the pattern "at <y, x>"
        matches = re.findall(r'<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>', subgoal)
        if matches:
            bbox = [[int(float(match[0])/1000*self.image_size[1]), int(float(match[1])/1000*self.image_size[0])] for match in matches]
        else:
            bbox = []        
        response = re.sub(
            r'<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>',
            '<bbox>',
            subgoal
        )
        
        return response, bbox
    
    def update_history_subgoals(self, subgoal: str):
        if self.subgoal_type == "simple_subgoal":
            if self.history_simple_subgoals:
                if self.history_simple_subgoals[-1] != subgoal:
                    self.history_simple_subgoals.append(subgoal)
            else:
                self.history_simple_subgoals.append(subgoal)
        else:
            assistant_prompt, bbox = self._parse_grounded_subgoal(subgoal)
            if self.history_grounded_subgoals:
                if self.history_grounded_subgoals[-1] != assistant_prompt:
                    self.history_grounded_subgoals.append(assistant_prompt)
                    self.history_grounded_bboxes.extend(bbox)
            else:
                self.history_grounded_subgoals.append(assistant_prompt)
                self.history_grounded_bboxes.extend(bbox)
    
    def prepare_infer_request(
        self,
        image_query: np.ndarray,
        step_idx: int,
        reporter_result: bool | None = None,
    ) -> dict:
        
        image_path = os.path.join(self.save_dir, f"step_{step_idx}_image.png")
        if not os.path.exists(image_path):
            imageio.imwrite(image_path, image_query)
        history_subgoals = (
            self.history_simple_subgoals
            if self.subgoal_type == "simple_subgoal"
            else self.history_grounded_subgoals
        )
        messages = self.prompt.build_messages(
            subgoal_type=self.subgoal_type,
            task_goal=self.task_goal,
            history_subgoals=history_subgoals,
            reporter_result=reporter_result,
            has_video=self.video_path is not None,
        )
        
        infer_request_dict = {
            "messages": messages,
            "images": [image_path]
        }
        
        if self.video_path is not None:
            infer_request_dict["videos"] = [self.video_path]
        self.prompt.validate_media_alignment(
            messages,
            image_count=len(infer_request_dict["images"]),
            video_count=len(infer_request_dict.get("videos", [])),
        )
            
        if self.subgoal_type == "grounded_subgoal":
            infer_request_dict["objects"] = {"ref": [], "bbox": self.history_grounded_bboxes}
        
        print("\n\n")
        pprint.pprint(infer_request_dict)
        
        return InferRequest(**infer_request_dict)

    def call(
        self,
        image_query: np.ndarray,
        step_idx: int,
        keep_period: int = 0,
        reporter_result: bool | None = None,
    ) -> str:
        if step_idx <= keep_period and self.last_response is not None:
            # some tasks that require press button, qwen models always skip
            # add some hard-coded rules to fix it
            response = self.last_response
        else:
            infer_request = self.prepare_infer_request(
                image_query,
                step_idx,
                reporter_result,
            )
            response = self.engine.infer([infer_request], request_config=RequestConfig(max_tokens=128, temperature=0))
            response = response[0].choices[0].message.content
        
        subgoal, structured_response = parse_manager_response(response)
        self.last_verification_passed = None
        print("Manager raw response: ", response)
        if reporter_result is True and self.interaction_verification_enabled:
            reported_verification_passed = parse_verification_passed(
                structured_response
            )
            interaction_pairs_valid, interaction_task_class = (
                validate_interaction_pairs(
                    structured_response,
                    self.last_response,
                )
            )
            self.last_verification_passed = (
                reported_verification_passed is True
                and interaction_pairs_valid
            )
            if structured_response is None:
                verification_log = {
                    "interaction_pairs": [],
                    "verification_passed": reported_verification_passed,
                    "json_parse_success": False,
                }
            else:
                verification_log = {
                    "interaction_pairs": structured_response.get(
                        "interaction_pairs", []
                    ),
                    "verification_passed": reported_verification_passed,
                    "json_parse_success": True,
                }
            verification_log["interaction_task_class"] = interaction_task_class
            verification_log["interaction_pairs_valid"] = interaction_pairs_valid
            verification_log["verification_approved"] = (
                self.last_verification_passed
            )
            verification_log["init_frame_update_approved"] = (
                self.last_verification_passed is True
            )
            if (
                self.last_verification_passed is not True
                and self.last_response is not None
            ):
                # Failed, missing, or malformed verification cannot advance
                # either the executable subgoal or Reporter's init frame.
                subgoal = self.last_response
            print(
                "Manager interaction verification JSON: ",
                json.dumps(verification_log, ensure_ascii=False),
            )
        print("Manager executable subgoal: ", subgoal)
        self.last_response = subgoal
        self.update_history_subgoals(subgoal)
        return self._parse_subgoal_for_vla(subgoal)
