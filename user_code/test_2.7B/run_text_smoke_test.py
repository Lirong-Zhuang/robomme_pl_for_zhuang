"""Run a text-only smoke test against the local Qwen3.8 checkpoint.

This script deliberately sends no image field.  It uses the same ms-swift
TransformersEngine path as run_test.py so its output can distinguish a
language/FP8 loading problem from a vision-input problem.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path


DEFAULT_MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "/data/public/qwen/Qwen3.8-27B-FP8",
)
# Qwen3.8 reuses the Qwen3.5 model implementation in Transformers/ms-swift,
# but it has its own prompt/chat template. These are internal registry keys,
# not the checkpoint or model version.
SWIFT_MODEL_TYPE = "qwen3_5"
SWIFT_TEMPLATE_TYPE = "qwen3_8"
DEFAULT_PROMPT = (
    "Michael is Daniel's father, and Daniel is Emma's father. "
    "What is Michael's relationship to Emma? Explain your reasoning briefly "
    "in English, then end with 'Final answer: <one word>'."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the local Qwen3.8 checkpoint and run one text-only request "
            "without reading or encoding any images."
        )
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument(
        "--attn-impl",
        choices=("flash_attention_2", "sdpa"),
        default="flash_attention_2",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--inspect-fp8",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the first layer's MLP projection and FP8-scale metadata.",
    )
    return parser.parse_args()


def validate_qwen38_config(model_path: Path) -> None:
    """Confirm that the checkpoint uses Qwen3.8's registered architecture."""
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as source:
        config = json.load(source)

    model_type = config.get("model_type")
    architectures = config.get("architectures", [])
    if (
        model_type != SWIFT_MODEL_TYPE
        or "Qwen3_5ForConditionalGeneration" not in architectures
    ):
        raise ValueError(
            "This Qwen3.8 test expects config.json to declare "
            "model_type='qwen3_5' and "
            "Qwen3_5ForConditionalGeneration. "
            f"Found model_type={model_type!r}, architectures={architectures!r}."
        )

    quantization_config = config.get("quantization_config", {})
    exclusions = quantization_config.get("modules_to_not_convert", [])
    shadowing_exclusions = [
        item
        for item in exclusions
        if isinstance(item, str) and item.endswith(".mlp.gate")
    ]
    if shadowing_exclusions:
        raise ValueError(
            "This is the unpatched Qwen3.8 FP8 config: "
            f"{len(shadowing_exclusions)} '.mlp.gate' exclusions can shadow "
            "gate_proj and discard its FP8 scales. First run "
            "prepare_qwen38_fp8_model.py and pass its output directory to "
            "--model-path."
        )


def print_fp8_diagnostics(model: object) -> bool:
    """Print lightweight evidence about the loaded MLP projection modules."""
    backbone = getattr(model, "model", None)
    language_model = getattr(backbone, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None or len(layers) == 0:
        print("FP8 diagnostics: could not locate model.language_model.layers")
        return False

    mlp = getattr(layers[0], "mlp", None)
    if mlp is None:
        print("FP8 diagnostics: could not locate layer 0 MLP")
        return False

    print("\nFP8 diagnostics (language layer 0):")
    fp8_ready = True
    for projection_name in ("gate_proj", "up_proj", "down_proj"):
        projection = getattr(mlp, projection_name, None)
        if projection is None:
            print(f"  {projection_name}: missing")
            fp8_ready = False
            continue

        weight = getattr(projection, "weight", None)
        scale = getattr(projection, "weight_scale_inv", None)
        weight_dtype = getattr(weight, "dtype", None)
        scale_dtype = getattr(scale, "dtype", None)
        scale_shape = tuple(scale.shape) if scale is not None else None
        print(
            f"  {projection_name}: type={type(projection).__name__}, "
            f"weight_dtype={weight_dtype}, "
            f"weight_scale_inv_dtype={scale_dtype}, "
            f"weight_scale_inv_shape={scale_shape}"
        )
        if scale is None or "float8" not in str(weight_dtype):
            fp8_ready = False
    return fp8_ready


def main() -> None:
    args = parse_args()
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")
    if not args.prompt.strip():
        raise ValueError("--prompt must not be empty")

    # These variables must be set before importing swift/transformers.  USE_HF
    # selects Hugging Face as the runtime-kernel source; the model path below
    # remains a local directory and inference remains local.
    os.environ["USE_HF"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    try:
        from swift import get_model_processor
        from swift import get_template
        from swift.infer_engine import InferRequest
        from swift.infer_engine import RequestConfig
        from swift.infer_engine import TransformersEngine
    except ImportError as error:
        raise RuntimeError(
            "A recent ms-swift with Qwen3.8 support is required. "
            "Activate the robomme-qwen38 environment first."
        ) from error

    model_path = Path(args.model_path).expanduser().resolve()
    validate_qwen38_config(model_path)
    print("Checkpoint family: Qwen3.8")
    print(f"Model weights path: {model_path}")
    print(
        "Internal ms-swift architecture key: "
        f"{SWIFT_MODEL_TYPE} (Qwen3.8 reuses the Qwen3.5 implementation)"
    )
    print(f"Internal ms-swift template key: {SWIFT_TEMPLATE_TYPE}")
    print("Input modality: text only (no images are passed to InferRequest)")
    print(f"Prompt: {args.prompt}")

    load_started = time.perf_counter()
    model, processor = get_model_processor(
        str(model_path),
        model_type=SWIFT_MODEL_TYPE,
        attn_impl=args.attn_impl,
    )
    print(f"Model load time: {time.perf_counter() - load_started:.2f} s")

    if args.inspect_fp8:
        if not print_fp8_diagnostics(model):
            raise RuntimeError(
                "Qwen3.8 FP8 validation failed: gate_proj, up_proj and "
                "down_proj must all have float8 weights and weight_scale_inv."
            )

    template = get_template(
        processor,
        template_type=SWIFT_TEMPLATE_TYPE,
        enable_thinking=False,
        preserve_thinking=False,
    )
    engine = TransformersEngine(model, template=template, max_batch_size=1)
    request_config = RequestConfig(
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.0,
        seed=42,
    )

    # Intentionally omit images=... so this request never enters the visual
    # preprocessing path.
    request = InferRequest(
        messages=[{"role": "user", "content": args.prompt}],
    )
    inference_started = time.perf_counter()
    responses = engine.infer([request], request_config=request_config)
    elapsed = time.perf_counter() - inference_started
    if len(responses) != 1:
        raise RuntimeError(f"Expected one response, received {len(responses)}")

    content = responses[0].choices[0].message.content
    print("\nResponse:")
    print(content)
    print("\nResponse repr (reveals replacement characters explicitly):")
    print(repr(content))
    print(f"Inference time: {elapsed:.2f} s")

    normalized_content = re.sub(
        r"^<think>\s*</think>\s*",
        "",
        content.strip(),
        count=1,
    ).strip()
    print(f"Normalized answer: {normalized_content!r}")

    final_answer_match = re.search(
        r"Final\s+answer\s*:\s*([A-Za-z-]+)[.!]?\s*$",
        normalized_content,
        flags=re.IGNORECASE,
    )
    final_answer = final_answer_match.group(1).casefold() if final_answer_match else None
    print(f"Parsed final answer: {final_answer!r}")

    if final_answer == "grandfather":
        print(
            "Result: PASS - the text-only language path completed the "
            "relationship reasoning task correctly."
        )
    else:
        print(
            "Result: FAIL/INVESTIGATE - the text-only language path did not "
            "produce the expected deterministic answer."
        )


if __name__ == "__main__":
    main()
