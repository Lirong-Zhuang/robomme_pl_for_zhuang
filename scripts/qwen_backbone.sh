#!/usr/bin/env bash

# Print the local path for one supported Qwen3-VL model name. The third
# argument is an optional explicit path override.
resolve_qwen_backbone_path() {
    local model_name=$1
    local model_root=$2
    local model_path=${3:-}

    case "$model_name" in
        Qwen3-VL-4B-Instruct|Qwen3-VL-8B-Instruct)
            ;;
        *)
            echo "ERROR: Unsupported Qwen backbone name: $model_name" >&2
            echo "Supported names: Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct" >&2
            return 1
            ;;
    esac

    if [[ -n "$model_path" ]]; then
        printf '%s\n' "$model_path"
    else
        printf '%s/%s\n' "${model_root%/}" "$model_name"
    fi
}

# Backward-compatible convenience wrapper used by one-backbone training jobs.
resolve_qwen_backbone() {
    QWEN_BACKBONE_NAME="${QWEN_BACKBONE_NAME:-Qwen3-VL-4B-Instruct}"
    QWEN_BACKBONE_ROOT="${QWEN_BACKBONE_ROOT:-/data/public/qwen}"
    QWEN_BACKBONE_PATH=$(resolve_qwen_backbone_path \
        "$QWEN_BACKBONE_NAME" \
        "$QWEN_BACKBONE_ROOT" \
        "${QWEN_BACKBONE_PATH:-}")
    export QWEN_BACKBONE_NAME QWEN_BACKBONE_ROOT QWEN_BACKBONE_PATH
}
