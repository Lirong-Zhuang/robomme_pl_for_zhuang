"""Render the model-facing grasp contract from the trusted action catalog."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any


def coordinate_instruction(space: str) -> str:
    if space == "normalized_1000":
        return "Use [x,y] normalized to 0–1000 on current_front, with the origin at top left."
    if space == "pixel":
        return "Use [x,y] in original 256×256 pixels on current_front, with the origin at top left."
    raise ValueError(f"unsupported VLM point coordinate space: {space!r}")


def grasp_requirements(contract: Mapping[str, Any]) -> str:
    roles = []
    for target in contract["targets"]:
        colors = target.get("allowed_colors")
        if colors:
            color = "/".join(colors)
        elif target["color_hint"] == "required":
            color = "visible color"
        else:
            color = "null"
        roles.append(f"{target['role']}: {target['point_semantics']}; color_hint={color}")
    return " | ".join(roles)


def render_grounding_prompt(
    system_prompt: str,
    subgoals: list[dict[str, Any]],
    contract_for: Callable[[str], Mapping[str, Any]],
    coordinate_space: str,
) -> str:
    grasps = []
    for subgoal in subgoals:
        contract = contract_for(subgoal["action_name"])
        if contract["mode"] == "vlm_points":
            grasps.append((subgoal["subgoal_key"], contract))

    lines = [
        "VLM grounding",
        'For other keys and STOP, use {"status":"not_required","image_ref":null,"targets":{}}.'
        if grasps else
        'For every key, use {"status":"not_required","image_ref":null,"targets":{}}.',
    ]
    if grasps:
        lines.insert(1, "Ground the selected next_subgoal_key. Only these keys require a point:")
        lines[2:2] = [f"- {key}: {grasp_requirements(contract)}" for key, contract in grasps]
        key, contract = grasps[0]
        targets = {
            target["role"]: {
                "point_xy": ["x", "y"],
                "color_hint": (
                    (target.get("allowed_colors") or ["visible_color"])[0]
                    if target["color_hint"] == "required" else None
                ),
            }
            for target in contract["targets"]
        }
        example = json.dumps({
            "next_subgoal_key": key,
            "grounding": {"status": "ok", "image_ref": "current_front", "targets": targets},
        }, separators=(",", ":")).replace('["x","y"]', '[x,y]')
        lines.extend((
            coordinate_instruction(coordinate_space),
            f"Field example (replace x,y with numbers and use the selected target's color): {example}",
            'If unclear, use {"status":"ambiguous","image_ref":"current_front","targets":{}}.',
        ))

    block = "\n".join(lines)
    # Replace prose from existing task configs too, including stale coordinate rules.
    heading = r"(?im)^\s*(?:#{1,6}[ \t]+)?VLM grounding[ \t]*:?\s*\n"
    section = re.search(heading, system_prompt)
    if section is None:
        return f"{system_prompt.rstrip()}\n\n{block}"
    following = system_prompt[section.end():]
    output = re.search(r"(?im)^(?:#{1,6}[ \t]+)?Output(?: format)?[ \t]*:?[ \t]*$", following)
    if output is None:
        # Keep the phase instructions of a legacy prompt with a nonstandard heading.
        return f"{system_prompt.rstrip()}\n\n{block}\nUse this grounding block for the response."
    return f"{system_prompt[:section.start()].rstrip()}\n\n{block}\n\n{following[output.start():]}"


def render_followup_output(
    subgoals: list[dict[str, Any]],
    contract_for: Callable[[str], Mapping[str, Any]],
    *,
    uses_keyframes: bool,
) -> str:
    lines = [
        "Follow-up response",
        'Set top-level status to "completed", "incomplete", or "failed" for the executed action.',
        'Use "ok", "ambiguous", and "not_required" only in grounding.status for the next action.',
        "Use process and final frames. A visible release over the destination counts despite later occlusion.",
        "Use incomplete only when repeating is physically meaningful.",
    ]
    for subgoal in subgoals:
        if contract_for(subgoal["action_name"])["mode"] == "vlm_points":
            continue
        template = subgoal["template"]
        if any("{" in text for text in (template, *subgoal.get("acceptance", {}).values())):
            continue
        example = {
            "status": "completed",
            "description": "<visible evidence for the executed action>",
            "next_subgoal_key": subgoal["subgoal_key"],
            "next_subgoal": template,
            "arguments": {},
            "grounding": {"status": "not_required", "image_ref": None, "targets": {}},
        }
        if uses_keyframes:
            example["keyframe_save"] = None
        lines.extend((
            "Example only. Choose the actual status and next action from the observations.",
            json.dumps(example, ensure_ascii=False, separators=(",", ":")),
        ))
        break
    return "\n".join(lines)
