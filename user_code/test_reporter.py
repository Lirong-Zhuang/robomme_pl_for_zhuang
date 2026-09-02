"""Sequentially score one Reporter on a labelled Reporter test set.

Edit the USER CONFIG section below, then run:

    uv run python user_code/test_reporter.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

# ============================= USER CONFIG =============================
REPO_ROOT = Path(__file__).resolve().parents[1]

CUDA_VISIBLE_DEVICES = "0"

DEFAULT_REPORTER_MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_REPORTER_ADAPTER_PATH = "runs/ckpts/reporter/qwen_reporter_v4.1_simple_subgoal/v0-20260901-043235/checkpoint-900"

DEFAULT_TESTSET_PATH = "data/trinity_preprocessed_data/reporter_binfill_data_2"
DEFAULT_SUBGOAL_TYPE = "simple"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "reporter_evaluation"
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


def _resolve_repo_path(value: str) -> str:
    if not value or Path(value).is_absolute():
        return value
    return str((REPO_ROOT / value).resolve())


def _resolve_test_dataset(testset_path: Path, subgoal_type: str) -> Path:
    """Accept either a Reporter JSONL file or one of its parent directories."""
    path = testset_path.expanduser().resolve()
    if path.is_file():
        return path
    filenames = (
        f"{subgoal_type}_subgoal_test.jsonl",
        f"{subgoal_type}_subgoal_train.jsonl",
    )
    candidates = [path / filename for filename in filenames]
    candidates.extend(path / "reporter_qwenvl" / filename for filename in filenames)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find a {subgoal_type} Reporter JSONL under {path}. Tried:\n  {tried}"
    )


def _build_engine(model_path: str, adapter_path: str | None) -> tuple[Any, type, Any]:
    try:
        from swift.llm import InferRequest
        from swift.llm import PtEngine
        from swift.llm import RequestConfig
    except ImportError as error:
        raise RuntimeError(
            "ms-swift is required for Reporter scoring. Run this script in the same "
            "environment used by examples/robomme/eval.py."
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


def _print_summary(result: ReporterMetrics) -> None:
    print("\nSequential Reporter test summary")
    print(f"Dataset: {result.dataset_path}")
    print(f"Episodes: {result.episodes}")
    print(f"Unique frames: {result.total}")
    print(f"Duplicate rows skipped: {result.duplicates_skipped}")
    print(f"Correct: {result.correct}/{result.total}")
    print(f"Accuracy: {_format_rate(result.accuracy)}")
    print(f"Parse rate: {_format_rate(result.parse_rate)}")
    print(f"Complete recall: {_format_rate(result.completed_recall)}")
    print(f"Incomplete recall: {_format_rate(result.incomplete_recall)}")
    print("Invalid/unparseable JSON outputs count as incorrect.")


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
            "Reporter JSONL file, testset directory, or testset/reporter_qwenvl "
            "directory."
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
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = _resolve_test_dataset(args.testset_path, args.subgoal_type)
    adapter_path = "" if args.no_adapter else _resolve_repo_path(args.adapter_path)
    if adapter_path and not Path(adapter_path).is_dir():
        raise FileNotFoundError(f"Reporter adapter not found: {adapter_path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
            predictions_path=output_dir / "predictions.jsonl",
            image_root=args.image_root,
            max_samples=args.max_samples,
            progress_every=args.progress_every,
        )
    finally:
        if engine is not None:
            old_engine = engine
            engine = None
            del old_engine
            _clear_model_cache()

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _print_summary(result)
    print(f"Per-frame predictions: {output_dir / 'predictions.jsonl'}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
