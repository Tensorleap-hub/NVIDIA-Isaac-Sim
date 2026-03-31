#!/bin/bash
# Usage:
#   ./run_experiments.sh                               — runs the 5 hand-crafted experiment configs
#   ./run_experiments.sh experiments/experiment-1      — runs all *.yaml in the given directory
#   ./run_experiments.sh experiments/experiment-1 64   — same, but overrides num_frames to 64

# Path to Isaac Sim installation (contains python.sh)
ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/opt/IsaacSim}"

# All project paths derived from this script's location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPT="$SCRIPT_DIR/standalone_palletjack_sdg.py"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ── Parse arguments ────────────────────────────────────────────────────────────
N_SAMPLES_OVERRIDE=""
if [ -n "$2" ]; then
    N_SAMPLES_OVERRIDE="$2"
fi

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
[ -n "$N_SAMPLES_OVERRIDE" ] && echo "  n_samples  : $N_SAMPLES_OVERRIDE (override)"
echo ""

cd "$ISAAC_SIM_PATH"

for EXP_CONFIG in "${CONFIGS[@]}"; do
    EXP_NAME="$(basename "$EXP_CONFIG" .yaml)"
    OUTPUT_DIR="$BASE_OUTPUT/$EXP_NAME"
    mkdir -p "$OUTPUT_DIR"
    LOG_FILE="$OUTPUT_DIR/run.log"
    echo "--- Running $EXP_NAME ---"
    echo "    Log: $LOG_FILE"
    EXTRA_ARGS=""
    [ -n "$N_SAMPLES_OVERRIDE" ] && EXTRA_ARGS="--num_frames $N_SAMPLES_OVERRIDE"
    ./python.sh "$SCRIPT" --config "$EXP_CONFIG" --headless True --data_dir "$OUTPUT_DIR" $EXTRA_ARGS 2>&1 | tee "$LOG_FILE"
done

echo ""
echo "All experiments complete. Data at: $BASE_OUTPUT"
