"""Create a private Qwen3.8-27B-FP8 model view with a corrected config.

The official dense FP8 checkpoint contains per-layer ``*.mlp.gate`` entries in
``modules_to_not_convert``.  Transformers can match those entries against
``*.mlp.gate_proj``, leaving the quantized gate projections as ordinary BF16
Linear modules and dropping their ``weight_scale_inv`` tensors.

This utility does not copy the large checkpoint.  It creates symbolic links to
all source files and writes only a corrected config.json in the destination.
The shared source directory is never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("/data/public/qwen/Qwen3.8-27B-FP8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New private model directory to create; it must not already exist.",
    )
    return parser.parse_args()


def load_and_patch_config(config_path: Path) -> tuple[dict[str, Any], list[str]]:
    with config_path.open("r", encoding="utf-8") as source:
        config = json.load(source)

    quantization_config = config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        raise ValueError("config.json has no quantization_config object")

    exclusions = quantization_config.get("modules_to_not_convert")
    if not isinstance(exclusions, list) or not all(
        isinstance(item, str) for item in exclusions
    ):
        raise ValueError(
            "quantization_config.modules_to_not_convert is not a string list"
        )

    removed = [item for item in exclusions if item.endswith(".mlp.gate")]
    required_dense_entries = {
        f"model.language_model.layers.{layer_index}.mlp.gate"
        for layer_index in range(64)
    }
    missing_dense_entries = required_dense_entries.difference(removed)
    allowed_entries = required_dense_entries | {"mtp.layers.0.mlp.gate"}
    unexpected_entries = set(removed).difference(allowed_entries)
    if missing_dense_entries or unexpected_entries:
        raise ValueError(
            "The '.mlp.gate' exclusions do not match the expected 64 dense "
            "layers and optional MTP layer. "
            f"Missing={sorted(missing_dense_entries)}, "
            f"unexpected={sorted(unexpected_entries)}. "
            "Refusing to patch an unexpected checkpoint."
        )

    quantization_config["modules_to_not_convert"] = [
        item for item in exclusions if not item.endswith(".mlp.gate")
    ]
    return config, removed


def main() -> None:
    args = parse_args()
    source_dir = args.source.expanduser().resolve()
    output_dir = args.output.expanduser().absolute()

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source model directory does not exist: {source_dir}")
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}\n"
            "Choose a new directory so no existing files are overwritten."
        )
    if output_dir == source_dir:
        raise ValueError("--output must be different from --source")

    config, removed = load_and_patch_config(source_dir / "config.json")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    try:
        for source_item in source_dir.iterdir():
            destination_item = output_dir / source_item.name
            if source_item.name == "config.json":
                with destination_item.open("w", encoding="utf-8") as destination:
                    json.dump(config, destination, ensure_ascii=False, indent=2)
                    destination.write("\n")
            else:
                os.symlink(
                    source_item,
                    destination_item,
                    target_is_directory=source_item.is_dir(),
                )
    except Exception:
        shutil.rmtree(output_dir)
        raise

    print(f"Source model: {source_dir}")
    print(f"Fixed model view: {output_dir}")
    print(f"Removed exclusions: {len(removed)}")
    print("Shared source files modified: no")
    print("Large weight files copied: no (symbolic links only)")
    print("\nUse this path with --model-path:")
    print(output_dir)


if __name__ == "__main__":
    main()
