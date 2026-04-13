#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/opt/IsaacSim}"
VENV_DIR="${LOOP_VENV_DIR:-${REPO_ROOT}/.sim_loop_venv}"
PYTHON_BIN="${ISAAC_SIM_PATH}/kit/python/bin/python3"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Isaac Python not found at ${PYTHON_BIN}" >&2
    echo "Set ISAAC_SIM_PATH to your Isaac Sim install root." >&2
    exit 1
fi

echo "Creating loop venv at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

echo "Installing loop dependencies"
"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install -r "${REPO_ROOT}/local_requirements.txt"

echo ""
echo "Loop venv ready:"
echo "  ${VENV_DIR}/bin/python"
