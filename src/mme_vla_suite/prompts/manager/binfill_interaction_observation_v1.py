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
    "TARGET OBJECT FROM SUBGOAL: {target_cube}\n"
    "<image>Observe the current image and output exactly one "
    "interaction pair. The interaction pair is "
    "[[\"robot\",\"<relationship>\",\"{target_cube}\"]]. The subject and object "
    "are fixed. Replace only <relationship> with one concrete uppercase relation "
    "inferred from the current image, such as HOLDS, ABOVE, or NA. Never output zero "
    "or multiple pairs. Output exactly one minified JSON line: "
    "{{\"interaction_pairs\":[[\"robot\",\"<relationship>\","
    "\"{target_cube}\"]]}}."
)

PUT_IN_BIN_TEMPLATE = (
    "{video_prefix}SUBGOAL TO VERIFY: {subgoal_to_verify}\n"
    "TARGET OBJECT FROM PREVIOUS SUBGOAL: {target_cube}\n"
    "<image>Observe only this subgoal in the current image and output exactly one "
    "interaction pair. The interaction pair is "
    "[[\"{target_cube}\",\"<relationship>\",\"bin\"]]. The subject and object are "
    "fixed. Replace only <relationship> with one concrete uppercase relation inferred "
    "from the current image, such as IN, OUTSIDE, ABOVE, or NA. Never output zero or "
    "multiple pairs. Output exactly one minified JSON line: "
    "{{\"interaction_pairs\":[[\"{target_cube}\",\"<relationship>\","
    "\"bin\"]]}}."
)

PRESS_BUTTON_TEMPLATE = (
    "{video_prefix}SUBGOAL TO VERIFY: {subgoal_to_verify}\n"
    "<image>Observe only this subgoal in the current image and output exactly one "
    "interaction pair. The interaction pair is "
    "[[\"robot\",\"<relationship>\",\"button\"]]. The subject and object are fixed. "
    "Replace only <relationship> with one concrete uppercase relation inferred from "
    "the current image, such as PRESSES, ABOVE, or NA. Never output zero or multiple "
    "pairs. Output exactly one minified JSON line: "
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
    target_cube = "unknown cube"

    if "pick up" in normalized_subgoal:
        color = resolve_expected_cube_color(subgoal_to_verify)
        if color is not None:
            target_cube = f"{color} cube"
        template = PICK_UP_TEMPLATE
    elif "put" in normalized_subgoal and "bin" in normalized_subgoal:
        previous_subgoals = history_subgoals
        if (
            previous_subgoals
            and " ".join(previous_subgoals[-1].lower().split())
            == normalized_subgoal
        ):
            previous_subgoals = previous_subgoals[:-1]
        color = resolve_expected_cube_color(
            "",
            previous_subgoals,
        )
        if color is not None:
            target_cube = f"{color} cube"
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
