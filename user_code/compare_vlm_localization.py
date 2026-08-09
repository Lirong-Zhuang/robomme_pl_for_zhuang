#!/usr/bin/env python3
"""Compare QwenVL localization coordinates against MemER for one episode.

The script parses ``Response:`` lines in the human-readable RoboMME ``.log``
files, aligns SwingXtimes predictions by semantic stage, and reports the
distance between QwenVL and MemER coordinates.  MemER is treated as the
reference, but this is a relative comparison rather than ground-truth error.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path


BOX_RE = re.compile(
    r"<\|box_start\|>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*<\|box_end\|>"
)
RESPONSE_RE = re.compile(r"^Response:\s*(.+)$")


@dataclass(frozen=True)
class Prediction:
    stage: str
    text: str
    y: int
    x: int


def stage_from_text(text: str) -> str | None:
    """Map a SwingXtimes response to a stable semantic stage."""
    normalized = " ".join(text.lower().split())
    if re.search(r"\bpick up the .+? cube\b", normalized):
        return "pickup-cube"
    if "right-side target" in normalized:
        if "second time" in normalized:
            return "right-target-2"
        return "right-target-1"
    if "left-side target" in normalized:
        if "second time" in normalized:
            return "left-target-2"
        return "left-target-1"
    if "press the button" in normalized:
        return "press-button"
    return None


def parse_log(path: Path) -> list[Prediction]:
    predictions: list[Prediction] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            response_match = RESPONSE_RE.match(line.strip())
            if not response_match:
                continue
            response = response_match.group(1)
            box_match = BOX_RE.search(response)
            stage = stage_from_text(response)
            if box_match is None or stage is None:
                continue
            predictions.append(
                Prediction(
                    stage=stage,
                    text=response,
                    y=int(box_match.group(1)),
                    x=int(box_match.group(2)),
                )
            )
    if not predictions:
        raise ValueError(f"No coordinate-bearing SwingXtimes responses found in {path}")
    return predictions


def select_by_stage(
    predictions: list[Prediction], sample: str
) -> dict[str, Prediction]:
    grouped: dict[str, list[Prediction]] = {}
    for prediction in predictions:
        grouped.setdefault(prediction.stage, []).append(prediction)

    selected: dict[str, Prediction] = {}
    for stage, values in grouped.items():
        if sample == "first":
            selected[stage] = values[0]
        elif sample == "last":
            selected[stage] = values[-1]
        else:
            selected[stage] = Prediction(
                stage=stage,
                text=f"mean of {len(values)} predictions",
                y=round(sum(value.y for value in values) / len(values)),
                x=round(sum(value.x for value in values) / len(values)),
            )
    return selected


def default_log_paths(runs_dir: Path, episode: int) -> tuple[Path, Path]:
    root = (
        runs_dir
        / "evaluation"
        / "symbolic-grounded-subgoal"
        / "ckpt79999"
        / "seed7"
    )
    qwen = root / "qwenvl" / "SwingXtimes" / "logs" / f"SwingXtimes_ep{episode}.log"
    memer = root / "memer" / "SwingXtimes" / "logs" / f"SwingXtimes_ep{episode}.log"
    return qwen, memer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare QwenVL and MemER coordinates for one SwingXtimes episode."
    )
    parser.add_argument("--episode", "--ep", type=int, required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--qwen-log", type=Path, help="Override the QwenVL .log path")
    parser.add_argument("--memer-log", type=Path, help="Override the MemER .log path")
    parser.add_argument(
        "--sample",
        choices=("first", "last", "mean"),
        default="first",
        help="Which repeated prediction to use within each semantic stage",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/localization_comparison")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_qwen, default_memer = default_log_paths(args.runs_dir, args.episode)
    qwen_log = args.qwen_log or default_qwen
    memer_log = args.memer_log or default_memer

    for path in (qwen_log, memer_log):
        if not path.is_file():
            raise FileNotFoundError(f"Log file not found: {path}")

    qwen = select_by_stage(parse_log(qwen_log), args.sample)
    memer = select_by_stage(parse_log(memer_log), args.sample)
    stage_order = [
        "pickup-cube",
        "right-target-1",
        "left-target-1",
        "right-target-2",
        "left-target-2",
        "press-button",
    ]

    rows: list[dict[str, object]] = []
    coordinate_diagonal = math.hypot(1000, 1000)
    for stage in stage_order:
        if stage not in qwen or stage not in memer:
            print(f"WARNING: skipping {stage}: missing from one or both logs")
            continue
        qwen_point = qwen[stage]
        memer_point = memer[stage]
        dy = qwen_point.y - memer_point.y
        dx = qwen_point.x - memer_point.x
        distance = math.hypot(dy, dx)
        rows.append(
            {
                "stage": stage,
                "qwen_y": qwen_point.y,
                "qwen_x": qwen_point.x,
                "memer_y": memer_point.y,
                "memer_x": memer_point.x,
                "delta_y": dy,
                "delta_x": dx,
                "distance": distance,
                "difference_percent": distance / coordinate_diagonal * 100,
            }
        )

    if not rows:
        raise RuntimeError("The two logs have no comparable coordinate-bearing stages")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"SwingXtimes_ep{args.episode}_{args.sample}"
    csv_path = args.output_dir / f"{stem}.csv"
    png_path = args.output_dir / f"{stem}.png"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Episode {args.episode}; sample={args.sample}; MemER is the reference")
    print("Percent = Euclidean coordinate distance / diagonal(1000 x 1000) * 100")
    print(f"{'stage':<20} {'QwenVL (y,x)':<18} {'MemER (y,x)':<18} {'distance':>10} {'percent':>10}")
    for row in rows:
        qwen_text = f"({row['qwen_y']},{row['qwen_x']})"
        memer_text = f"({row['memer_y']},{row['memer_x']})"
        print(
            f"{row['stage']:<20} {qwen_text:<18} {memer_text:<18} "
            f"{row['distance']:>10.2f} {row['difference_percent']:>9.2f}%"
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            f"Wrote {csv_path}, but plotting requires matplotlib: "
            "python -m pip install matplotlib"
        ) from exc

    labels = [str(row["stage"]) for row in rows]
    percentages = [float(row["difference_percent"]) for row in rows]
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(range(len(rows)), percentages, marker="o", linewidth=2)
    axis.set_xticks(range(len(rows)), labels, rotation=25, ha="right")
    axis.set_ylabel("QwenVL–MemER coordinate difference (%)")
    axis.set_xlabel("Semantic stage (execution order)")
    axis.set_title(f"SwingXtimes episode {args.episode}: localization difference")
    axis.grid(True, alpha=0.3)
    for index, percentage in enumerate(percentages):
        axis.annotate(f"{percentage:.2f}%", (index, percentage), xytext=(0, 7),
                      textcoords="offset points", ha="center")
    figure.tight_layout()
    figure.savefig(png_path, dpi=180)
    plt.close(figure)

    print(f"CSV:  {csv_path}")
    print(f"Plot: {png_path}")


if __name__ == "__main__":
    main()
