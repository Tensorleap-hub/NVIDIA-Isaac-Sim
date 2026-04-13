#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec "${REPO_ROOT}/simulation_calibration_loop/run_with_loop_venv.sh" \
    "${REPO_ROOT}/run_dinov2_optuna_loop.py" \
    "$@"
