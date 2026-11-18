#!/usr/bin/env bash

# Run the no-action BinFill reset -> simulator GT mask -> F2G diagnostic.

set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

GROUNDING_TEST_SCRIPT="$REPO_ROOT/src/harness/tests/test_grounding_env_gt.py" \
GROUNDING_TEST_LOG_STEM="test_grounding_env_gt" \
bash "$REPO_ROOT/scripts/test_grounding_pos.sh"
