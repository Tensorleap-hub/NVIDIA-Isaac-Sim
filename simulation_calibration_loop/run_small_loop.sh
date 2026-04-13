#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec "${REPO_ROOT}/simulation_calibration_loop/run_with_loop_venv.sh" \
    -m simulation_calibration_loop.test_isaac_small_loop \
    --config "${REPO_ROOT}/simulation_calibration_loop/test_isaac_small_loop.yaml" \
    "$@"
