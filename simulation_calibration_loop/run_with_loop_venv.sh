#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${LOOP_VENV_DIR:-${REPO_ROOT}/.sim_loop_venv}"
PYTHON_BIN="${VENV_DIR}/bin/python"
ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/opt/IsaacSim}"
NVJITLINK_LIB_DIR="${ISAAC_SIM_PATH}/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/nvjitlink/lib"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Loop venv missing at ${VENV_DIR}" >&2
    echo "Run: bash simulation_calibration_loop/setup_loop_venv.sh" >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -d "${NVJITLINK_LIB_DIR}" ]]; then
    export LD_LIBRARY_PATH="${NVJITLINK_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec "${PYTHON_BIN}" "$@"
