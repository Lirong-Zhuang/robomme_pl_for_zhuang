"""Subgoal-specific interaction-observation prompts for BinFill."""

from __future__ import annotations

from mme_vla_suite.interaction_verification import resolve_expected_cube_color


VERSION = "binfill_interaction_observation_v1"

SYSTEM_PROMPT = (
    "You infer one concrete interaction relation for one robot subgoal from the "
    "current observation."
)

PICK_UP_TEMPLATE = (
    "{video_prefix}SUBGOAL TO VERIFY: {subgoal_to_verify}\n"
    "<image>Observe the current image and output exactly one "
    "interaction pair. The interaction pair is "
    "[[\"robot\",\"<relationship>\",\"<color> cube\"]]. Never output zero or "
    "multiple pairs. Replace <color> with the cube color stated in SUBGOAL TO "
    "VERIFY. Replace <relationship> with one concrete uppercase relation inferred "
    "only from the current image, such as HOLDS, ABOVE, or NA. Output exactly one "
    "minified JSON line: "
    "{{\"interaction_pairs\":[[\"robot\",\"<relationship>\","
    "\"<color> cube\"]]}}."
)

PUT_IN_BIN_TEMPLATE = (
    "{video_prefix}SUBGOAL TO VERIFY: {subgoal_to_verify}\n"
    "<image>Observe only this subgoal in the current image and output exactly one "
    "interaction pair. The interaction pair is "
    "[[\"<color> cube\",\"<relationship>\",\"bin\"]]. Never output zero or "
    "multiple pairs. Replace <color> with the cube color stated in SUBGOAL TO "
    "VERIFY. Replace <relationship> with one concrete uppercase relation "
    "inferred only from the current image, such as IN, OUTSIDE, ABOVE, or NA. Output "
    "exactly one minified JSON line: "
    "{{\"interaction_pairs\":[[\"<color> cube\",\"<relationship>\","
    "\"bin\"]]}}."
)

PRESS_BUTTON_TEMPLATE = (
    "{video_prefix}SUBGOAL TO VERIFY: {subgoal_to_verify}\n"
    "<image>Observe only this subgoal in the current image and output exactly one "
    "interaction pair. The interaction pair is "
    "[[\"robot\",\"<relationship>\",\"button\"]]. Never output zero or multiple "
    "pairs. Replace "
    "<relationship> with one concrete uppercase relation inferred from the current "
    "image only, such as PRESSES, ABOVE, or NA. Output exactly one minified JSON "
    "line: "
    "{{\"interaction_pairs\":[[\"robot\",\"<relationship>\",\"button\"]]}}."
)

PROMPT_SOURCE = "\0".join(
    (
        SYSTEM_PROMPT,
        VERSION,
        PICK_UP_TEMPLATE,
        PUT_IN_BIN_TEMPLATE,
        PRESS_BUTTON_TEMPLATE,
    )
)


def build_user_prompt(
    *,
    subgoal_to_verify: str,
    history_subgoals: list[str],
    video_prefix: str,
    task_goal: str,
) -> str:
    """Select exactly one BinFill observation template from subgoal keywords."""

    del task_goal  # BinFill verification prompts do not use the overall task.
    normalized_subgoal = " ".join(subgoal_to_verify.lower().split())
    target_cube = ""

    if "pick up" in normalized_subgoal:
        template = PICK_UP_TEMPLATE
    elif "put" in normalized_subgoal and "bin" in normalized_subgoal:
        color = resolve_expected_cube_color(
            subgoal_to_verify,
            history_subgoals,
        )
        target_cube = f"{color} cube" if color is not None else "unknown"
        template = PUT_IN_BIN_TEMPLATE
    elif "press" in normalized_subgoal and "button" in normalized_subgoal:
        template = PRESS_BUTTON_TEMPLATE
    else:
        raise ValueError(
            "BinFill interaction observation supports only 'pick up', "
            "'put ... bin', and 'press ... button' subgoals; got "
            f"{subgoal_to_verify!r}"
        )

    return template.format(
        video_prefix=video_prefix,
        subgoal_to_verify=subgoal_to_verify,
        target_cube=target_cube,
    )
