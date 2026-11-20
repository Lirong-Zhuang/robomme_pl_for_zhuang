#!/usr/bin/env bash

# Run the no-action BinFill reset -> simulator GT mask -> F2G diagnostic.

set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

# Native square resolution of the independent grounding camera.
# Change this value to 256, 512, 768, 1024, etc. before launching the test.
GROUNDING_CAMERA_RESOLUTION="${GROUNDING_CAMERA_RESOLUTION:-512}"

GROUNDING_CAMERA_RESOLUTION="$GROUNDING_CAMERA_RESOLUTION" \
GROUNDING_TEST_SCRIPT="$REPO_ROOT/src/harness/tests/test_grounding_env_gt.py" \
GROUNDING_RUN_NAME="grounding_env_gt" \
bash "$REPO_ROOT/scripts/test_grounding_pos.sh"
