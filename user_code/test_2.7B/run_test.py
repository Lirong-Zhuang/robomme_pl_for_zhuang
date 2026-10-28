"""Run the joint Qwen 2.7B subgoal-and-action feasibility test."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from prompts import match_action_name
from runtime_metrics import GpuMemoryMonitor
from runtime_metrics import timing_summary

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "test_2.7B"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs" / "test_2.7B"

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mme_vla_suite.reporter_evaluation import score_reporter_completion

DEFAULT_MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "/data/public/qwen/Qwen3.8-27B-FP8",
)


def _directory_name(value: str) -> str:
    name = value.strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise argparse.ArgumentTypeError(
            "name must be one non-empty directory name without slashes"
        )
    return name


def _parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    candidate = fenced.group(1) if fenced else stripped
    if not candidate.startswith("{"):
        embedded = re.search(r"\{.*\}", candidate, re.DOTALL)
        candidate = embedded.group(0) if embedded else candidate
    try:
        value = json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).casefold().strip().split())


def _load_rows(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("messages"), list) or not isinstance(row.get("images"), list):
                raise ValueError(f"Invalid dataset row at {path}:{line_number}")
            rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"Dataset contains no rows: {path}")
    return rows


def _expected_from_row(row: dict[str, Any]) -> dict[str, Any]:
    assistants = [
        message for message in row["messages"] if message.get("role") == "assistant"
    ]
    if len(assistants) != 1:
        raise ValueError("Every test row must contain exactly one assistant label")
    expected = _parse_json_object(assistants[0].get("content", ""))
    if expected is None:
        raise ValueError("Assistant label is not valid JSON")
    return expected


def _input_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [message for message in row["messages"] if message.get("role") != "assistant"]


def _build_engine(
    model_path: str,
    adapter_path: str | None,
    attn_impl: str,
):
    try:
        from swift import get_model_processor
        from swift import get_template
        from swift.infer_engine import InferRequest
        from swift.infer_engine import RequestConfig
        from swift.infer_engine import TransformersEngine
    except ImportError as error:
        raise RuntimeError(
            "A recent ms-swift with Qwen3.8 support is required. Activate the "
            "server inference environment or update ms-swift."
        ) from error

    model, processor = get_model_processor(model_path, attn_impl=attn_impl)
    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as error:
            raise RuntimeError("peft is required when --adapter-path is used") from error
        model = PeftModel.from_pretrained(model, adapter_path)

    # Qwen3.8 is a hybrid thinking model. Build the template explicitly so
    # behavior does not depend on the installed ms-swift version's default.
    template = get_template(
        processor,
        enable_thinking=False,
        preserve_thinking=False,
    )
    if getattr(template, "enable_thinking", None) is not False:
        raise RuntimeError("Failed to disable Qwen thinking mode")
    engine = TransformersEngine(model, template=template, max_batch_size=1)
    return engine, InferRequest, RequestConfig


def _score(
    expected: dict[str, Any],
    predicted: dict[str, Any] | None,
    task: str,
) -> dict[str, bool]:
    completed_correct = (
        predicted is not None
        and isinstance(predicted.get("completed"), bool)
        and predicted["completed"] == expected["completed"]
    )
    subgoal_correct = (
        predicted is not None
        and "next_subgoal" in predicted
        and _normalize_text(predicted["next_subgoal"])
        == _normalize_text(expected["next_subgoal"])
    )
    action_correct = (
        predicted is not None
        and "action_name" in predicted
        and _normalize_text(predicted["action_name"])
        == _normalize_text(expected["action_name"])
    )
    pair_consistent = False
    if predicted is not None and "next_subgoal" in predicted and "action_name" in predicted:
        predicted_subgoal = _normalize_text(predicted["next_subgoal"])
        if predicted_subgoal == "stop":
            pair_consistent = predicted["action_name"] is None
        else:
            matched_action = match_action_name(task, str(predicted["next_subgoal"]))
            pair_consistent = (
                matched_action is not None
                and _normalize_text(predicted["action_name"])
                == _normalize_text(matched_action)
            )
    return {
        "parsed": predicted is not None,
        "completed_correct": completed_correct,
        "subgoal_correct": subgoal_correct,
        "action_correct": action_correct,
        "pair_consistent": pair_consistent,
        "joint_correct": completed_correct and subgoal_correct and action_correct,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name",
        type=_directory_name,
        default="binfill_test",
        help="Dataset directory name under --data-root (default: binfill_test).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Dataset parent directory (default: data/test_2.7B).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Explicit JSONL path; otherwise uses "
            "data/test_2.7B/<dataset-name>/subgoal_action_test.jsonl."
        ),
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument(
        "--attn-impl",
        choices=("flash_attention_2", "sdpa"),
        default="flash_attention_2",
        help=(
            "Attention backend (default: flash_attention_2 for lower latency; "
            "use sdpa if FlashAttention is unavailable)."
        ),
    )
    parser.add_argument(
        "--result-name",
        type=_directory_name,
        required=True,
        help=(
            "Required run name; results are written to "
            "runs/test_2.7B/<result-name>."
        ),
    )
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Must remain 1 so each row has an exact end-to-end latency.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--image-max-token-num",
        type=int,
        default=256,
        help="Maximum visual tokens per image (default: 256).",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--early-tolerance-calls",
        type=int,
        default=1,
        help="Reporter-style allowed early completion calls (default: 1).",
    )
    parser.add_argument(
        "--full-credit-delay-calls",
        type=int,
        default=1,
        help="Reporter-style full-credit delay calls (default: 1).",
    )
    parser.add_argument(
        "--maximum-delay-calls",
        type=int,
        default=2,
        help="Reporter-style maximum delay before timeout (default: 2).",
    )
    parser.add_argument(
        "--reporter-debounce",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Debounce continuous completed=true outputs like Reporter test.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the first request without loading a model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("--batch-size must be 1 for exact per-step VLM timing")
    if args.early_tolerance_calls < 0:
        raise ValueError("--early-tolerance-calls must be non-negative")
    if args.full_credit_delay_calls < 0:
        raise ValueError("--full-credit-delay-calls must be non-negative")
    if args.maximum_delay_calls < args.full_credit_delay_calls:
        raise ValueError(
            "--maximum-delay-calls must be at least --full-credit-delay-calls"
        )
    if args.image_max_token_num < 1:
        raise ValueError("--image-max-token-num must be at least 1")
    dataset_path = (
        args.dataset.expanduser().resolve()
        if args.dataset is not None
        else (
            args.data_root.expanduser().resolve()
            / args.dataset_name
            / "subgoal_action_test.jsonl"
        )
    )
    rows = _load_rows(dataset_path, args.max_samples)
    if args.dry_run:
        preview = {
            "mode": "subgoal_action",
            "dataset_name": args.dataset_name,
            "dataset": str(dataset_path),
            "sample_count": len(rows),
            "messages": _input_messages(rows[0]),
            "images": rows[0]["images"],
            "expected": _expected_from_row(rows[0]),
            "metadata": rows[0].get("metadata", {}),
            "inference_config": {
                "enable_thinking": False,
                "attn_impl": args.attn_impl,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "image_max_token_num": args.image_max_token_num,
            },
        }
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return
    if not args.model_path:
        raise ValueError("Set --model-path or the QWEN_MODEL_PATH environment variable")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["IMAGE_MAX_TOKEN_NUM"] = str(args.image_max_token_num)
    os.environ.pop("PYTORCH_NO_CUDA_MEMORY_CACHING", None)
    gpu_monitor = GpuMemoryMonitor(args.cuda_visible_devices)
    model_load_started = time.perf_counter()
    engine, InferRequest, RequestConfig = _build_engine(
        str(Path(args.model_path).expanduser().resolve()),
        str(Path(args.adapter_path).expanduser().resolve()) if args.adapter_path else None,
        args.attn_impl,
    )
    model_load_seconds = time.perf_counter() - model_load_started
    gpu_monitor.mark_model_loaded()
    request_config = RequestConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.0,
        seed=42,
    )
    result_dir = DEFAULT_RUN_ROOT / args.result_name
    result_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = result_dir / "predictions.jsonl"
    episodes_path = result_dir / "buffered_completion_episodes.jsonl"

    totals = {
        "samples": 0,
        "parsed": 0,
        "completed_correct": 0,
        "subgoal_correct": 0,
        "action_correct": 0,
        "pair_consistent": 0,
        "joint_correct": 0,
    }
    completion_records: list[dict[str, Any]] = []
    step_times_ms: list[float] = []
    with predictions_path.open("w", encoding="utf-8") as output:
        for row_index, row in enumerate(rows, start=1):
            inference_started = time.perf_counter()
            request_messages = _input_messages(row)
            request_images = list(row["images"])
            request = InferRequest(
                messages=request_messages,
                images=request_images,
            )
            responses = engine.infer([request], request_config=request_config)
            elapsed_ms = (time.perf_counter() - inference_started) * 1000
            if len(responses) != 1:
                raise RuntimeError("Model did not return exactly one response")
            step_times_ms.append(elapsed_ms)
            gpu_monitor.sample()

            raw_response = responses[0].choices[0].message.content
            predicted = _parse_json_object(raw_response)
            expected = _expected_from_row(row)
            metadata = row.get("metadata", {})
            task = str(metadata.get("task", "BinFill"))
            scores = _score(expected, predicted, task)
            totals["samples"] += 1
            for key, value in scores.items():
                totals[key] += int(value)
            timing = {
                "vlm_response_ms": round(elapsed_ms, 3),
                "image_count": len(request_images),
                "measurement_scope": (
                    "InferRequest construction with image paths through "
                    "engine.infer returning output"
                ),
            }
            record = {
                "test_step": row_index,
                "metadata": metadata,
                "inference_config": {
                    "enable_thinking": False,
                    "attn_impl": args.attn_impl,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_k": 1,
                    "top_p": 1.0,
                    "image_max_token_num": int(os.environ["IMAGE_MAX_TOKEN_NUM"]),
                },
                "input": {
                    "messages": request_messages,
                    "images": request_images,
                },
                "expected": expected,
                "output": {
                    "raw_response": raw_response,
                    "parsed": predicted,
                },
                "predicted": predicted,
                "raw_response": raw_response,
                "timing": timing,
                **scores,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

            if metadata.get("call_type") == "followup":
                predicted_completed = (
                    predicted.get("completed") if predicted is not None else None
                )
                completion_records.append(
                    {
                        "task": task,
                        "episode": int(metadata["episode"]),
                        "current_step": int(metadata["current_step"]),
                        "line_number": row_index,
                        "expected": expected.get("completed"),
                        "predicted": (
                            predicted_completed
                            if isinstance(predicted_completed, bool)
                            else None
                        ),
                    }
                )
            print(
                f"Processed {row_index}/{len(rows)} | "
                f"VLM response {elapsed_ms:.1f} ms"
            )

    completion_report = score_reporter_completion(
        completion_records,
        early_tolerance_calls=args.early_tolerance_calls,
        full_credit_delay_calls=args.full_credit_delay_calls,
        maximum_delay_calls=args.maximum_delay_calls,
        reporter_debounce=args.reporter_debounce,
    )
    with episodes_path.open("w", encoding="utf-8") as output:
        for episode in completion_report["episodes"]:
            output.write(json.dumps(episode, ensure_ascii=False) + "\n")

    denominator = totals["samples"]
    summary = {
        "mode": "subgoal_action",
        "dataset_name": args.dataset_name,
        "result_name": args.result_name,
        "dataset": str(dataset_path),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "adapter_path": (
            str(Path(args.adapter_path).expanduser().resolve())
            if args.adapter_path
            else None
        ),
        "inference_config": {
            "enable_thinking": False,
            "preserve_thinking": False,
            "attn_impl": args.attn_impl,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_k": 1,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "image_max_token_num": int(os.environ["IMAGE_MAX_TOKEN_NUM"]),
            "cuda_memory_caching": True,
        },
        **totals,
        "parse_rate": totals["parsed"] / denominator,
        "completed_accuracy": totals["completed_correct"] / denominator,
        "subgoal_accuracy": totals["subgoal_correct"] / denominator,
        "action_accuracy": totals["action_correct"] / denominator,
        "pair_consistency": totals["pair_consistent"] / denominator,
        "joint_accuracy": totals["joint_correct"] / denominator,
        "buffered_completion": {
            **{
                key: value
                for key, value in completion_report.items()
                if key != "episodes"
            },
            "note": (
                "Uses the Reporter causal tolerance scorer on followup rows. "
                "This is a temporal completion diagnostic; strict next_subgoal "
                "and action_name accuracy remain the primary text metrics."
            ),
            "episode_details": str(episodes_path),
        },
        "gpu_and_timing": {
            "gpu": gpu_monitor.summary(),
            "timing": timing_summary(
                step_times_ms,
                model_load_seconds=model_load_seconds,
            ),
        },
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    del engine
    gc.collect()
    gpu_monitor.close()


if __name__ == "__main__":
    main()

