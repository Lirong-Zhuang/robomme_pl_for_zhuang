"""Resolve the supported local Qwen3-VL backbones by model name."""

import os
from pathlib import Path


DEFAULT_QWEN_BACKBONE_NAME = "Qwen3-VL-4B-Instruct"
DEFAULT_QWEN_BACKBONE_ROOT = "/data/public/qwen"
SUPPORTED_QWEN_BACKBONES = (
    "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-8B-Instruct",
)


def resolve_qwen_backbone_path(
    model_name: str | None = None,
    *,
    model_root: str | os.PathLike[str] | None = None,
    model_path: str | os.PathLike[str] | None = None,
) -> str:
    """Return a local backbone path, validating the selected model name.

    ``model_path`` is an optional explicit override for installations whose
    directory layout differs from ``<model_root>/<model_name>``.
    """
    resolved_name = model_name or os.environ.get(
        "QWEN_BACKBONE_NAME", DEFAULT_QWEN_BACKBONE_NAME
    )
    if resolved_name not in SUPPORTED_QWEN_BACKBONES:
        supported = ", ".join(SUPPORTED_QWEN_BACKBONES)
        raise ValueError(
            f"Unsupported Qwen backbone {resolved_name!r}. Supported names: {supported}"
        )

    if model_path:
        return str(Path(model_path).expanduser())

    resolved_root = model_root or os.environ.get(
        "QWEN_BACKBONE_ROOT", DEFAULT_QWEN_BACKBONE_ROOT
    )
    return str(Path(resolved_root).expanduser() / resolved_name)
