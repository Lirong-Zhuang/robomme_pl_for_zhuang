from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness.grounding_prompt import render_followup_output
from harness.grounding_prompt import render_grounding_prompt
from harness.init_planning import build_init_planning_messages
from harness.init_planning import validate_task_config
from harness.prompts import build_task_init_messages
from harness.prompts.init_prompt import DIRECT_INIT_PROMPT_VERSION


class DirectInitPlanningPromptTests(unittest.TestCase):
    def test_binfill_followup_separates_verification_from_grounding(self) -> None:
        resources = Path(__file__).resolve().parents[1] / "resources"
        catalog = json.loads((resources / "action_candidates" / "action_catalog.json").read_text())
        config = json.loads((resources / "task_config_examples" / "task_goal_only.json").read_text())
        contracts = {action["name"]: action["perception"] for action in catalog["actions"]}
        rendered = render_followup_output(config["subgoals"], contracts.__getitem__, uses_keyframes=False)

        self.assertIn('top-level status to "completed", "incomplete", or "failed"', rendered)
        self.assertIn('Use "ok", "ambiguous", and "not_required" only in grounding.status', rendered)
        example = json.loads(rendered.splitlines()[-1])
        self.assertEqual(example["next_subgoal_key"], "put_into_bin")
        self.assertEqual(example["status"], "completed")
        self.assertEqual(example["grounding"], {"status": "not_required", "image_ref": None, "targets": {}})

    def test_followup_example_keeps_keyframe_field(self) -> None:
        rendered = render_followup_output(
            [{"subgoal_key": "press", "template": "press the first button", "action_name": "solve_button"}],
            lambda _: {"mode": "none"},
            uses_keyframes=True,
        )
        self.assertIsNone(json.loads(rendered.splitlines()[-1])["keyframe_save"])

    def test_runtime_replaces_old_points_and_only_lists_grasp_keys(self) -> None:
        resources = Path(__file__).resolve().parents[1] / "resources"
        catalog = json.loads((resources / "action_candidates" / "action_catalog.json").read_text())
        contracts = {action["name"]: action["perception"] for action in catalog["actions"]}
        subgoals = [
            {"subgoal_key": "hold_end", "action_name": "grasp_and_lift_peg_side"},
            {"subgoal_key": "insert", "action_name": "insert_peg"},
        ]
        old = "Verify and advance.\n\nVLM grounding\nUse 256 pixels for insert.\n\nOutput\nReturn JSON only."
        rendered = render_grounding_prompt(old, subgoals, contracts.__getitem__, "normalized_1000")
        self.assertIn("hold_end: peg:", rendered)
        self.assertIn("black/white", rendered)
        self.assertIn("not_required", rendered)
        self.assertIn("0–1000", rendered)
        self.assertIn("Return JSON only.", rendered)
        self.assertNotIn("256", rendered)
        self.assertNotIn("insert", rendered)

    def test_runtime_omits_coordinate_instructions_for_path_only_task(self) -> None:
        rendered = render_grounding_prompt(
            "VLM grounding\nOld grid rule.\n\nOutput\nReturn JSON only.",
            [{"subgoal_key": "route", "action_name": "solve_swingonto_with_direction"}],
            lambda _: {"mode": "none", "targets": []},
            "pixel",
        )
        self.assertIn("For every key", rendered)
        self.assertNotIn("point_xy", rendered)
        self.assertNotIn("Old grid rule", rendered)

    def test_buttonunmaskswap_example_uses_fixed_button_subgoals(self) -> None:
        resources = (
            Path(__file__).resolve().parents[1] / "resources"
        )
        catalog = json.loads(
            (resources / "action_candidates" / "action_catalog.json").read_text()
        )
        config = json.loads(
            (
                resources
                / "task_config_examples"
                / "task_goal_plus_keyframes.json"
            ).read_text()
        )

        validated = validate_task_config(
            config,
            catalog,
            task_name="ButtonUnmaskSwap",
        )
        button_subgoals = [
            item
            for item in validated["subgoals"]
            if item["action_name"] == "solve_button"
        ]
        self.assertEqual(
            [item["subgoal_key"] for item in button_subgoals],
            ["press_first_button", "press_second_button"],
        )
        self.assertTrue(all("{" not in item["template"] for item in button_subgoals))

    def test_direct_prompt_has_no_tool_transport_instructions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            messages, _ = build_init_planning_messages(
                task_name="BinFill",
                output_path=Path(directory) / "BinFill.json",
            )

        system = messages[0]["content"]
        user = messages[1]["content"]
        self.assertNotIn("Use the provided read tool", system)
        self.assertNotIn("call the provided write tool", system)
        self.assertNotIn("Available tools:", user)
        self.assertIn("Return the complete task_config.v1 JSON", system)
        self.assertIn("No filesystem write tool is available", system)

    def test_direct_prompt_embeds_all_required_resources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            messages, _ = build_init_planning_messages(
                task_name="BinFill",
                output_path=Path(directory) / "BinFill.json",
            )

        user = messages[1]["content"]
        self.assertEqual(user.count("<read_result path="), 5)
        self.assertIn("action_catalog.v2", user)
        self.assertIn("task_goal_only.json", user)
        self.assertIn("task_goal_plus_video_subgoal_list.json", user)
        self.assertIn("task_goal_plus_keyframes.json", user)

    def test_direct_prompt_version_is_distinct(self) -> None:
        self.assertTrue(DIRECT_INIT_PROMPT_VERSION.endswith("_direct_json_v1"))

    def test_tool_prompt_transport_remains_available(self) -> None:
        messages = build_task_init_messages(
            task_name="BinFill",
            action_candidates_path="/resources/action_catalog.json",
            task_config_examples_index_path="/resources/README.md",
            task_config_output_path="/output/BinFill.json",
        )

        self.assertIn("Use the provided read tool", messages[0]["content"])
        self.assertIn("write tool exactly once", messages[0]["content"])
        self.assertIn("Available tools:", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
