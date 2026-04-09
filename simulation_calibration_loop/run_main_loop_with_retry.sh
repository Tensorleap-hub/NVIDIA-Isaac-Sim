#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-60}"

attempt=1
while true; do
    echo "[retry] starting attempt ${attempt}" >&2
    if "${REPO_ROOT}/simulation_calibration_loop/run_main_loop.sh" "$@"; then
        echo "[retry] workflow completed successfully" >&2
        exit 0
    else
        status=$?
    fi

    echo "[retry] workflow failed with exit code ${status}; retrying in ${SLEEP_SECONDS}s" >&2
    sleep "${SLEEP_SECONDS}"
    attempt=$((attempt + 1))
done
