"""Qwen3.8-27B subgoal-and-point inference for live Oracle Plus rollouts."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Qwen3.8 FP8 needs the Hugging Face runtime-kernel path used by the validated
# offline feasibility runner. Set these before importing swift/transformers.
os.environ["USE_HF"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

SWIFT_MODEL_TYPE = "qwen3_5"
SWIFT_TEMPLATE_TYPE = "qwen3_8"
INFERENCE_SEED = 42
JSON_RESPONSE_PREFIX = "{"


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first complete JSON object, tolerating wrapper text or fences."""
    if not isinstance(text, str):
        return None
    object_start = text.find("{")
    if object_start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[object_start:])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def validate_qwen38_config(model_path: Path) -> None:
    """Require the patched Qwen3.8 FP8 configuration used by run_test.py."""
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config does not exist: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = config.get("model_type")
    architectures = config.get("architectures", [])
    if (
        model_type != SWIFT_MODEL_TYPE
        or "Qwen3_5ForConditionalGeneration" not in architectures
    ):
        raise ValueError(
            "Oracle Plus expects the Qwen3.8 checkpoint to declare "
            "model_type='qwen3_5' and Qwen3_5ForConditionalGeneration. "
            f"Found model_type={model_type!r}, architectures={architectures!r}."
        )
    exclusions = config.get("quantization_config", {}).get(
        "modules_to_not_convert", []
    )
    shadowing = [
        item
        for item in exclusions
        if isinstance(item, str) and item.endswith(".mlp.gate")
    ]
    if shadowing:
        raise ValueError(
            "The Qwen3.8 FP8 config is not patched: '.mlp.gate' exclusions "
            "can shadow gate_proj scales. Run "
            "user_code/test_27B/prepare_qwen38_fp8_model.py first."
        )


@dataclass(frozen=True)
class SubgoalDecision:
    """Validated VLM output containing no robot function name."""

    completed: bool
    next_subgoal: str
    point: tuple[int, int] | None = None
    description: str | None = None
    subgoal_sequence: tuple[str, ...] = ()
    strategy: str | None = None
    evidence: dict[str, Any] | None = None

    @property
    def is_stop(self) -> bool:
        normalized = self.next_subgoal.strip().casefold()
        return normalized == "stop" or normalized.startswith(
            ("stop ", "stop,", "stop.")
        )

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
        *,
        include_description: bool,
    ) -> "SubgoalDecision":
        forbidden_fields = sorted(
            field for field in ("action_name", "action_sequence") if field in value
        )
        if forbidden_fields:
            raise ValueError(
                "27B output must not contain action fields: "
                f"{forbidden_fields}"
            )
        completed = value.get("completed")
        if not isinstance(completed, bool):
            raise ValueError("27B output field 'completed' must be a boolean")
        next_subgoal = value.get("next_subgoal")
        if not isinstance(next_subgoal, str) or not next_subgoal.strip():
            raise ValueError("27B output field 'next_subgoal' must be non-empty text")
        next_subgoal = next_subgoal.strip()

        point_value = value.get("point")
        if point_value is None:
            point = None
        else:
            if (
                not isinstance(point_value, (list, tuple))
                or len(point_value) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, (int, float))
                    for item in point_value
                )
            ):
                raise ValueError("point must be null or point=[y, x] with two numbers")
            y, x = float(point_value[0]), float(point_value[1])
            if not math.isfinite(y) or not math.isfinite(x) or y < 0 or x < 0:
                raise ValueError("point=[y, x] must contain finite non-negative pixels")
            point = (int(round(y)), int(round(x)))

        if include_description:
            description = value.get("description")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(
                    "The description-enabled prompt requires a non-empty description"
                )
            description = description.strip()
        else:
            if "description" in value:
                raise ValueError(
                    "The no-description prompt forbids the description field"
                )
            description = None

        subgoals = value.get("subgoal_sequence", [])
        if not isinstance(subgoals, list) or not all(
            isinstance(item, str) and item.strip() for item in subgoals
        ):
            raise ValueError("subgoal_sequence must be a list of non-empty strings")
        strategy_value = value.get("strategy")
        if strategy_value is None:
            strategy = None
        elif not isinstance(strategy_value, str) or not strategy_value.strip():
            raise ValueError("strategy must be non-empty text when provided")
        else:
            strategy = strategy_value.strip()
        evidence_value = value.get("evidence")
        if evidence_value is None:
            evidence = None
        elif not isinstance(evidence_value, dict) or not all(
            isinstance(key, str) for key in evidence_value
        ):
            raise ValueError("evidence must be an object with string keys")
        else:
            evidence = dict(evidence_value)

        return cls(
            completed=completed,
            next_subgoal=next_subgoal,
            point=point,
            description=description,
            subgoal_sequence=tuple(item.strip() for item in subgoals),
            strategy=strategy,
            evidence=evidence,
        )


class VLMSubgoalPlanner27B:
    """Load Qwen3.8-27B once and produce subgoal-and-point decisions."""

    def __init__(
        self,
        *,
        model_path: str,
        attn_impl: str = "flash_attention_2",
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        try:
            from swift import get_model_processor
            from swift import get_template
            from swift.infer_engine import InferRequest
            from swift.infer_engine import RequestConfig
            from swift.infer_engine import TransformersEngine
        except ImportError as error:
            raise RuntimeError(
                "A recent ms-swift with Qwen3.8 support is required. Run "
                "Oracle Plus in the robomme-qwen38 environment."
            ) from error

        resolved_model_path = Path(model_path).expanduser().resolve()
        validate_qwen38_config(resolved_model_path)
        model, processor = get_model_processor(
            str(resolved_model_path),
            model_type=SWIFT_MODEL_TYPE,
            attn_impl=attn_impl,
        )
        template = get_template(
            processor,
            template_type=SWIFT_TEMPLATE_TYPE,
            enable_thinking=False,
            preserve_thinking=False,
            # Qwen3.8 normally inserts an empty <think></think> block when
            # thinking is disabled. An explicit response prefix takes priority
            # and makes the visible answer start directly with a JSON object.
            response_prefix=JSON_RESPONSE_PREFIX,
        )
        if getattr(template, "enable_thinking", None) is not False:
            raise RuntimeError("Failed to disable Qwen3.8 thinking mode")
        if getattr(template, "response_prefix", None) != JSON_RESPONSE_PREFIX:
            raise RuntimeError("Failed to force direct Qwen3.8 JSON output")
        self.engine = TransformersEngine(model, template=template, max_batch_size=1)
        self.InferRequest = InferRequest
        self.request_config = RequestConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.0,
            seed=INFERENCE_SEED,
        )
        self.model_path = str(resolved_model_path)
        self.attn_impl = attn_impl
        self.inference_seed = INFERENCE_SEED
        self.response_prefix = JSON_RESPONSE_PREFIX

    def infer(
        self,
        *,
        messages: list[dict[str, str]],
        images: list[str],
        include_description: bool,
    ) -> tuple[SubgoalDecision, str, float, float]:
        placeholder_count = sum(
            message.get("content", "").count("<image>") for message in messages
        )
        if placeholder_count != len(images):
            raise ValueError(
                f"Prompt/image mismatch: {placeholder_count} placeholders for "
                f"{len(images)} images"
            )
        started_at = time.perf_counter()
        request = self.InferRequest(messages=messages, images=images)
        responses = self.engine.infer(
            [request],
            request_config=self.request_config,
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        if len(responses) != 1:
            raise RuntimeError(f"Expected one 27B response, received {len(responses)}")
        raw_response = responses[0].choices[0].message.content
        parsed = parse_json_object(raw_response)
        if parsed is None:
            raise ValueError(f"27B response is not a JSON object: {raw_response!r}")
        decision = SubgoalDecision.from_mapping(
            parsed,
            include_description=include_description,
        )
        request_to_decision_ms = (time.perf_counter() - started_at) * 1000
        return decision, raw_response, elapsed_ms, request_to_decision_ms
