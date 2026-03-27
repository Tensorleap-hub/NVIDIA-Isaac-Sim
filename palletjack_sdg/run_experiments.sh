#!/bin/bash
# Usage:
#   ./run_experiments.sh                        — runs the 5 hand-crafted experiment configs
#   ./run_experiments.sh experiments/experiment-1  — runs all *.yaml in the given directory

# Path to Isaac Sim installation (contains python.sh)
ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/opt/IsaacSim}"

# All project paths derived from this script's location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPT="$SCRIPT_DIR/standalone_palletjack_sdg.py"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ── Resolve config directory ───────────────────────────────────────────────────
if [ -n "$1" ]; then
    # Argument can be absolute or relative to SCRIPT_DIR
    if [[ "$1" = /* ]]; then
        CONFIG_DIR="$1"
    else
        CONFIG_DIR="$SCRIPT_DIR/$1"
    fi
    RUN_NAME="$(basename "$CONFIG_DIR")"
else
    CONFIG_DIR="$SCRIPT_DIR/experiments"
    RUN_NAME="experiments"
fi

if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: directory not found: $CONFIG_DIR"
    exit 1
fi

CONFIGS=("$CONFIG_DIR"/*.yaml)
if [ ${#CONFIGS[@]} -eq 0 ] || [ ! -f "${CONFIGS[0]}" ]; then
    echo "Error: no *.yaml files found in $CONFIG_DIR"
    exit 1
fi

BASE_OUTPUT="$SCRIPT_DIR/palletjack_data/${RUN_NAME}_${TIMESTAMP}"

echo "Starting experiments"
echo "  Config dir : $CONFIG_DIR"
echo "  Output root: $BASE_OUTPUT"
echo "  Isaac Sim  : $ISAAC_SIM_PATH"
echo "  Configs    : ${#CONFIGS[@]}"
echo ""

cd "$ISAAC_SIM_PATH"

for EXP_CONFIG in "${CONFIGS[@]}"; do
    EXP_NAME="$(basename "$EXP_CONFIG" .yaml)"
    OUTPUT_DIR="$BASE_OUTPUT/$EXP_NAME"
    echo "--- Running $EXP_NAME ---"
    ./python.sh "$SCRIPT" --config "$EXP_CONFIG" --headless True --data_dir "$OUTPUT_DIR"
done

echo ""
echo "All experiments complete. Data at: $BASE_OUTPUT"
