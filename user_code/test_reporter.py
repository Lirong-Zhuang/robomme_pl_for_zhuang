"""Sequentially score one Reporter on a labelled Reporter test set.

Edit the USER CONFIG section below, then run:

    micromamba activate robomme
    python user_code/test_reporter.py --result-name my_reporter_test
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

# ============================= USER CONFIG =============================
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

CUDA_VISIBLE_DEVICES = "0"

DEFAULT_REPORTER_MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_REPORTER_ADAPTER_PATH = "runs/ckpts/reporter/qwen_reporter_v4.1_simple_subgoal/v0-20260901-043235/checkpoint-900"

DEFAULT_TESTSET_PATH = "data/trinity_preprocessed_data/reporter_binfill_data_2"
DEFAULT_SUBGOAL_TYPE = "simple"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "reporter_evaluation"
DEFAULT_RESULT_NAME = "reporter_qwen_v4.1_ckpt900"
# ======================================================================

# This must be set before swift/torch loads the model. If "1" is selected,
# that physical GPU is exposed inside the process as cuda:0.
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# Match the resource limits used by examples/robomme/eval.py.
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "256")
os.environ.setdefault("VIDEO_MAX_TOKEN_NUM", "64")
os.environ.setdefault("FPS_MAX_FRAMES", "10")

from mme_vla_suite.reporter_evaluation import ReporterMetrics
from mme_vla_suite.reporter_evaluation import evaluate_reporter_sequence
from mme_vla_suite.reporter_evaluation import resolve_reporter_test_dataset
from mme_vla_suite.reporter_evaluation import score_reporter_completion


def _resolve_repo_path(value: str) -> str:
    if not value or Path(value).is_absolute():
        return value
    return str((REPO_ROOT / value).resolve())


def _build_engine(model_path: str, adapter_path: str | None) -> tuple[Any, type, Any]:
    try:
        from swift.llm import InferRequest
        from swift.llm import PtEngine
        from swift.llm import RequestConfig
    except ImportError as error:
        raise RuntimeError(
            "ms-swift is not installed in the current Python environment. "
            "Activate the same environment used for Reporter training and "
            "examples/robomme/eval.py, then run: "
            "`micromamba activate robomme` followed by "
            "`python user_code/test_reporter.py`. Do not use `uv run` unless "
            "ms-swift is also installed in the uv environment."
        ) from error

    engine_kwargs: dict[str, Any] = {
        "model_id_or_path": model_path,
        "attn_impl": "sdpa",
    }
    if adapter_path:
        engine_kwargs["adapters"] = [adapter_path]
    print(
        f"Loading Reporter model {model_path}"
        + (f" with adapter {adapter_path}" if adapter_path else " without an adapter")
    )
    engine = PtEngine(**engine_kwargs)
    return engine, InferRequest, RequestConfig(max_tokens=64, temperature=0)


def _clear_model_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _format_rate(rate: float | None) -> str:
    return "n/a" if rate is None else f"{100 * rate:.2f}%"


def _result_name(value: str) -> str:
    """Require one user-selected directory name below the output root."""
    name = value.strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise argparse.ArgumentTypeError(
            "result name must be one non-empty directory name without slashes"
        )
    return name


def _error_type(expected: bool, predicted: bool | None) -> str:
    if predicted is None:
        return "invalid_output"
    if expected:
        return "false_negative"
    return "false_positive"


def _export_errors(
    prediction_records: list[dict[str, Any]],
    errors_path: Path,
    error_frames_dir: Path,
) -> dict[str, Any]:
    """Export incorrect predictions and copy both images used for each decision."""
    if error_frames_dir.exists():
        shutil.rmtree(error_frames_dir)
    error_frames_dir.mkdir(parents=True)

    errors: list[dict[str, Any]] = []
    incomparable_calls_skipped = 0
    for record in prediction_records:
        if not record.get("label_comparable", True):
            incomparable_calls_skipped += 1
            continue
        if record["correct"]:
            continue

        error_index = len(errors) + 1
        error_type = _error_type(record["expected"], record["predicted"])
        prefix = (
            f"error{error_index:04d}_{record['task']}_ep{record['episode']}_"
            f"step{record['current_step']}_{error_type}"
        )
        used_init_source = Path(record["used_init_image"])
        current_source = Path(record["current_image"])
        used_init_target = error_frames_dir / (
            f"{prefix}_used_init{used_init_source.suffix}"
        )
        current_target = error_frames_dir / (
            f"{prefix}_current{current_source.suffix}"
        )
        shutil.copy2(used_init_source, used_init_target)
        shutil.copy2(current_source, current_target)

        error_record = dict(record)
        error_record.update(
            {
                "error_index": error_index,
                "error_type": error_type,
                "dataset_init_frame_name": Path(
                    record["dataset_init_image"]
                ).name,
                "used_init_frame_name": used_init_source.name,
                "error_frame_name": current_source.name,
                "saved_used_init_image": str(
                    used_init_target.relative_to(errors_path.parent)
                ),
                "saved_error_image": str(
                    current_target.relative_to(errors_path.parent)
                ),
            }
        )
        errors.append(error_record)

    error_report = {
        "total_errors": len(errors),
        "incomparable_calls_skipped": incomparable_calls_skipped,
        "false_positives": sum(
            error["error_type"] == "false_positive" for error in errors
        ),
        "false_negatives": sum(
            error["error_type"] == "false_negative" for error in errors
        ),
        "invalid_outputs": sum(
            error["error_type"] == "invalid_output" for error in errors
        ),
        "errors": errors,
    }
    errors_path.write_text(
        json.dumps(error_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return error_report


def _print_completion_summary(completion_report: dict[str, Any]) -> None:
    print("\nTask-progress completion summary")
    print(f"Episodes: {completion_report['episode_count']}")
    print(
        "Mean completion: "
        f"{_format_rate(completion_report['mean_completion'])}"
    )
    print(
        "Fully completed episodes: "
        f"{completion_report['fully_completed_episodes']}/"
        f"{completion_report['episode_count']}"
    )
    print(
        "Premature-trigger episodes: "
        f"{completion_report['premature_trigger_episodes']}"
    )
    print(f"Stalled episodes: {completion_report['stalled_episodes']}")
    print(
        "Episodes with 3-4 call delay warnings: "
        f"{completion_report['episodes_with_delay_warning']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one Reporter frame by frame while using its own success "
            "predictions to update the next init frame."
        )
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_REPORTER_MODEL_PATH,
        help="Reporter base model path.",
    )
    parser.add_argument(
        "--adapter-path",
        default=str(DEFAULT_REPORTER_ADAPTER_PATH),
        help="Reporter LoRA adapter/checkpoint path.",
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Evaluate the base model without an adapter.",
    )
    parser.add_argument(
        "--testset-path",
        type=Path,
        default=DEFAULT_TESTSET_PATH,
        help=(
            "Reporter JSONL file, dataset root, testset directory, or "
            "testset/reporter_qwenvl directory."
        ),
    )
    parser.add_argument(
        "--subgoal-type",
        choices=("simple", "grounded"),
        default=DEFAULT_SUBGOAL_TYPE,
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Fallback image directory when JSONL paths came from another machine.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Parent directory that contains named Reporter result folders.",
    )
    parser.add_argument(
        "--result-name",
        type=_result_name,
        default=DEFAULT_RESULT_NAME,
        help="Name of this result folder below --output-dir.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--early-tolerance-calls",
        type=int,
        default=2,
        help="Maximum Reporter calls that a true prediction may precede its expected event.",
    )
    parser.add_argument(
        "--full-credit-delay-calls",
        type=int,
        default=2,
        help="Maximum delayed Reporter calls that receive full completion credit.",
    )
    parser.add_argument(
        "--maximum-delay-calls",
        type=int,
        default=4,
        help="Maximum delayed Reporter calls before the episode is considered stalled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.early_tolerance_calls < 0:
        raise ValueError("--early-tolerance-calls must be non-negative")
    if args.full_credit_delay_calls < 0:
        raise ValueError("--full-credit-delay-calls must be non-negative")
    if args.maximum_delay_calls < args.full_credit_delay_calls:
        raise ValueError(
            "--maximum-delay-calls must be at least --full-credit-delay-calls"
        )
    dataset_path = resolve_reporter_test_dataset(args.testset_path, args.subgoal_type)
    adapter_path = "" if args.no_adapter else _resolve_repo_path(args.adapter_path)
    if adapter_path and not Path(adapter_path).is_dir():
        raise FileNotFoundError(f"Reporter adapter not found: {adapter_path}")

    output_dir = args.output_dir.expanduser().resolve() / args.result_name
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    episodes_path = output_dir / "episode_completion.jsonl"
    errors_path = output_dir / "errors.json"
    error_frames_dir = output_dir / "error_frames"
    prediction_records: list[dict[str, Any]] = []
    engine = None
    try:
        engine, infer_request_type, request_config = _build_engine(
            args.model_path,
            adapter_path,
        )
        result = evaluate_reporter_sequence(
            name=args.subgoal_type,
            dataset_path=dataset_path,
            engine=engine,
            infer_request_type=infer_request_type,
            request_config=request_config,
            model_path=args.model_path,
            adapter_path=adapter_path,
            predictions_path=predictions_path,
            image_root=args.image_root,
            max_samples=args.max_samples,
            progress_every=args.progress_every,
            prediction_records_out=prediction_records,
        )
    finally:
        if engine is not None:
            old_engine = engine
            engine = None
            del old_engine
            _clear_model_cache()

    error_report = _export_errors(
        prediction_records,
        errors_path,
        error_frames_dir,
    )
    completion_report = score_reporter_completion(
        prediction_records,
        early_tolerance_calls=args.early_tolerance_calls,
        full_credit_delay_calls=args.full_credit_delay_calls,
        maximum_delay_calls=args.maximum_delay_calls,
    )
    with episodes_path.open("w", encoding="utf-8") as episodes_file:
        for episode in completion_report["episodes"]:
            output_episode = {
                "task": episode["task"],
                "episode": episode["episode"],
                "completion": episode["completion"],
                "completed_subgoal": (
                    f"{episode['completed_subgoals']}/{episode['total_subgoals']}"
                ),
                "status": episode["status"],
                "termination_frame": episode["termination_frame"],
            }
            episodes_file.write(json.dumps(output_episode, ensure_ascii=False) + "\n")
    summary_path = output_dir / "summary.json"
    summary = {
        "name": result.name,
        "dataset_path": result.dataset_path,
        "model_path": result.model_path,
        "adapter_path": result.adapter_path,
        "completion": {
            key: value
            for key, value in completion_report.items()
            if key != "episodes"
        },
        "frame_diagnostics": {
            "note": "Only calls whose prediction-driven subgoal matches the dataset subgoal are comparable.",
            "comparable_total": result.total,
            "incomparable_calls": sum(
                not record.get("label_comparable", True)
                for record in prediction_records
            ),
            "all_invalid_outputs": sum(
                record.get("predicted") is None for record in prediction_records
            ),
            "correct": result.correct,
            "parsed": result.parsed,
            "invalid": result.invalid,
            "accuracy": result.accuracy,
            "true_sample_correct": result.true_positive,
            "true_sample_total": result.completed_total,
            "true_sample_accuracy": result.completed_recall,
            "false_sample_correct": result.true_negative,
            "false_sample_total": result.incomplete_total,
            "false_sample_accuracy": result.incomplete_recall,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _print_completion_summary(completion_report)
    print(
        "Comparable-frame parsing diagnostic: "
        f"{result.parsed}/{result.total} valid outputs "
        f"({_format_rate(result.parse_rate)}); frame accuracy is not used for scoring"
    )
    print(f"Results directory: {output_dir}")
    print(f"Per-frame predictions: {predictions_path}")
    print(f"Per-episode completion details: {episodes_path}")
    print(f"Summary: {summary_path}")
    print(f"Errors: {errors_path} ({error_report['total_errors']} rows)")
    print(f"Error frame pairs: {error_frames_dir}")


if __name__ == "__main__":
    main()
