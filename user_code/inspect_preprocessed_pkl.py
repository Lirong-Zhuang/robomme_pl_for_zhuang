#!/usr/bin/env python3
"""Inspect one preprocessed RoboMME VLA PKL sample and export its images."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Preprocessed dataset root containing data/, features/, and meta/",
    )
    parser.add_argument("--sample", type=int, default=0, help="PKL sample ID")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/pkl_inspection"),
        help="Directory in which to save the report and images",
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


def save_image(value: Any, path: Path) -> None:
    image = np.asarray(value)
    image = np.squeeze(image)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"Expected HWC RGB/RGBA image, got {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.min() >= -1 and image.max() <= 1:
            image = (image + 1) / 2 if image.min() < 0 else image
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    iio.imwrite(path, image)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    pkl_path = dataset_dir / "data" / f"{args.sample}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL sample not found: {pkl_path}")

    output_dir = args.output_dir.expanduser().resolve() / f"sample_{args.sample}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with pkl_path.open("rb") as file:
        sample = pickle.load(file)
    if not isinstance(sample, dict):
        raise TypeError(f"Expected a dict in {pkl_path}, got {type(sample)}")

    image_keys = {
        "image": "front_rgb.png",
        "wrist_image": "wrist_rgb.png",
    }
    report: dict[str, Any] = {
        "pkl_file": str(pkl_path),
        "fields": {},
    }
    terminal_lines = [f"PKL file: {pkl_path}"]

    for key, value in sample.items():
        array = np.asarray(value) if isinstance(value, (np.ndarray, list, tuple)) else None
        field: dict[str, Any] = {"type": type(value).__name__}
        if array is not None:
            field.update({"shape": list(array.shape), "dtype": str(array.dtype)})

        if key in image_keys:
            image_path = output_dir / image_keys[key]
            save_image(value, image_path)
            np.save(output_dir / f"{key}.npy", np.asarray(value))
            field["image_file"] = str(image_path)
            field["raw_array_file"] = str(output_dir / f"{key}.npy")
            terminal_lines.append(
                f"{key}: shape={array.shape}, dtype={array.dtype}\n"
                f"  image: {image_path}"
            )
        elif array is not None and array.size > 100:
            raw_path = output_dir / f"{key}.npy"
            np.save(raw_path, array)
            field["raw_array_file"] = str(raw_path)
            terminal_lines.append(
                f"{key}: shape={array.shape}, dtype={array.dtype}\n"
                f"  raw: {raw_path}"
            )
        else:
            decoded = json_safe(value)
            field["value"] = decoded
            terminal_lines.append(f"{key}: {decoded!r}")

        report["fields"][key] = field

    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    terminal_lines.append(f"Report: {report_path}")
    print("\n".join(terminal_lines))


if __name__ == "__main__":
    main()
