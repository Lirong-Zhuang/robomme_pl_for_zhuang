#!/usr/bin/env python3
"""Inspect one RoboMME HDF5 episode/timestep and export its visual data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import imageio.v3 as iio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print and export every dataset under one HDF5 episode/timestep."
    )
    parser.add_argument("h5_path", type=Path, help="Path to the .h5 file")
    parser.add_argument("--episode", type=int, default=0, help="Episode index (default: 0)")
    parser.add_argument("--timestep", type=int, default=0, help="Timestep index (default: 0)")
    parser.add_argument(
        "--all-timesteps",
        action="store_true",
        help="Export every timestep in the selected episode; ignores --timestep",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/h5_inspection"),
        help="Directory for report, images, and large arrays",
    )
    parser.add_argument(
        "--max-print-elements",
        type=int,
        default=100,
        help="Print arrays up to this size in full; larger arrays are saved as .npy",
    )
    return parser.parse_args()


def decode_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return decode_scalar(value)


def safe_name(dataset_path: str) -> str:
    return dataset_path.strip("/").replace("/", "__")


def looks_like_image(array: np.ndarray) -> bool:
    return (
        array.ndim in (2, 3)
        and array.shape[0] > 8
        and array.shape[1] > 8
        and (array.ndim == 2 or array.shape[2] in (1, 3, 4))
    )


def save_visual(array: np.ndarray, output_path: Path) -> None:
    image = np.squeeze(array)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    if image.dtype == np.uint8:
        rendered = image
    else:
        finite = np.isfinite(image)
        if not finite.any():
            rendered = np.zeros(image.shape, dtype=np.uint8)
        else:
            low = float(image[finite].min())
            high = float(image[finite].max())
            if high > low:
                rendered = np.clip((image - low) / (high - low) * 255, 0, 255).astype(
                    np.uint8
                )
            else:
                rendered = np.zeros(image.shape, dtype=np.uint8)
    iio.imwrite(output_path, rendered)


def describe_dataset(
    logical_path: str,
    dataset: h5py.Dataset,
    output_dir: Path,
    max_print_elements: int,
    display_name: str | None = None,
) -> tuple[str, dict[str, Any]]:
    value = dataset[()]
    array = np.asarray(value)
    name = safe_name(logical_path)
    label = display_name or logical_path.rsplit("/", 1)[-1]
    details: dict[str, Any] = {
        "path": logical_path,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }

    if array.ndim == 0:
        decoded = decode_scalar(array.item())
        details["value"] = json_safe(decoded)
        return f"{label}: {decoded!r}", details

    if array.dtype.kind in {"S", "U", "O"}:
        decoded = json_safe(array)
        details["value"] = decoded
        return f"{label}: {decoded!r}", details

    if looks_like_image(array):
        image_dir = output_dir / "images"
        array_dir = output_dir / "arrays"
        image_dir.mkdir(parents=True, exist_ok=True)
        array_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{name}.png"
        raw_path = array_dir / f"{name}.npy"
        save_visual(array, image_path)
        np.save(raw_path, array)
        details.update(
            {
                "min": float(np.nanmin(array)),
                "max": float(np.nanmax(array)),
                "image_file": str(image_path),
                "raw_array_file": str(raw_path),
            }
        )
        return (
            f"{label}: shape={array.shape}, dtype={array.dtype}, "
            f"min={details['min']}, max={details['max']}\n"
            f"  image: {image_path}\n  raw:   {raw_path}",
            details,
        )

    if array.size <= max_print_elements:
        values = json_safe(array)
        details["value"] = values
        return f"{label}: {values}", details

    array_dir = output_dir / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    raw_path = array_dir / f"{name}.npy"
    np.save(raw_path, array)
    details.update(
        {
            "min": float(np.nanmin(array)),
            "max": float(np.nanmax(array)),
            "mean": float(np.nanmean(array)),
            "raw_array_file": str(raw_path),
        }
    )
    return (
        f"{label}: shape={array.shape}, dtype={array.dtype}, "
        f"min={details['min']}, max={details['max']}, mean={details['mean']}\n"
        f"  raw: {raw_path}",
        details,
    )


def collect_group(
    group: h5py.Group,
    prefix: str,
    output_dir: Path,
    max_print_elements: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    lines: list[str] = []
    records: list[dict[str, Any]] = []

    def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        logical_path = f"{prefix}/{name}" if name else prefix
        line, record = describe_dataset(
            logical_path,
            obj,
            output_dir,
            max_print_elements,
            display_name=name.rsplit("/", 1)[-1],
        )
        lines.append(line)
        records.append(record)

    group.visititems(visitor)
    return lines, records


def collect_timestep(
    timestep_group: h5py.Group,
    episode_key: str,
    timestep_key: str,
    output_dir: Path,
    max_print_elements: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    lines: list[str] = []
    records: list[dict[str, Any]] = []
    preferred_categories = ["obs", "action", "info"]
    remaining_categories = sorted(
        key for key in timestep_group.keys() if key not in preferred_categories
    )

    for category in preferred_categories + remaining_categories:
        if category not in timestep_group:
            continue
        lines.extend(["", f"--- {category.upper()} ---"])
        obj = timestep_group[category]
        if isinstance(obj, h5py.Group):
            category_lines, category_records = collect_group(
                obj,
                f"{episode_key}/{timestep_key}/{category}",
                output_dir,
                max_print_elements,
            )
        else:
            line, record = describe_dataset(
                f"{episode_key}/{timestep_key}/{category}",
                obj,
                output_dir,
                max_print_elements,
                display_name=category,
            )
            category_lines, category_records = [line], [record]
        lines.extend(category_lines)
        records.extend(category_records)

    return lines, records


def main() -> None:
    args = parse_args()
    h5_path = args.h5_path.expanduser().resolve()
    selection_dir = "all_timesteps" if args.all_timesteps else f"timestep_{args.timestep}"
    output_dir = (
        args.output_dir.expanduser().resolve()
        / f"episode_{args.episode}"
        / selection_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_key = f"episode_{args.episode}"
    timestep_key = f"timestep_{args.timestep}"

    with h5py.File(h5_path, "r") as h5_file:
        if episode_key not in h5_file:
            available = sorted(key for key in h5_file if key.startswith("episode_"))
            raise KeyError(f"{episode_key!r} not found. Available: {available[:20]}")

        episode = h5_file[episode_key]
        available_timesteps = sorted(
            (key for key in episode if key.startswith("timestep_")),
            key=lambda key: int(key.rsplit("_", 1)[-1]),
        )
        if args.all_timesteps:
            selected_timesteps = available_timesteps
        else:
            if timestep_key not in episode:
                raise KeyError(
                    f"{timestep_key!r} not found. Available: {available_timesteps[:20]}"
                )
            selected_timesteps = [timestep_key]

        setup_lines: list[str] = []
        setup_records: list[dict[str, Any]] = []
        if "setup" in episode:
            setup_lines, setup_records = collect_group(
                episode["setup"],
                f"{episode_key}/setup",
                output_dir,
                args.max_print_elements,
            )

        timestep_sections: list[str] = []
        timestep_records: dict[str, list[dict[str, Any]]] = {}
        for selected_key in selected_timesteps:
            timestep_index = int(selected_key.rsplit("_", 1)[-1])
            lines, records = collect_timestep(
                episode[selected_key],
                episode_key,
                selected_key,
                output_dir,
                args.max_print_elements,
            )
            timestep_sections.extend(
                ["", f"=== TIMESTEP {timestep_index} ===", *lines]
            )
            timestep_records[selected_key] = records

    header = [
        f"HDF5 file: {h5_path}",
        f"Episode: {episode_key}",
        (
            f"Timesteps: all ({len(selected_timesteps)})"
            if args.all_timesteps
            else f"Timestep: {timestep_key}"
        ),
        f"Output: {output_dir}",
        "",
        "=== EPISODE SETUP ===",
    ]
    report_lines = header + setup_lines + timestep_sections
    report = "\n".join(report_lines)

    report_path = output_dir / "report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {"setup": setup_records, "timesteps": timestep_records},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(report)
    print(f"\nSaved text report: {report_path}")
    print(f"Saved JSON summary: {summary_path}")


if __name__ == "__main__":
    main()
