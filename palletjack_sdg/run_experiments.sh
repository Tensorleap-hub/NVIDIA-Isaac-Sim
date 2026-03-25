#!/bin/bash

ISAAC_SIM_PATH='/isaac-sim'
SCRIPT="$ISAAC_SIM_PATH/palletjack_sdg/standalone_palletjack_sdg.py"
EXPERIMENTS_DIR="$ISAAC_SIM_PATH/palletjack_sdg/experiments"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_OUTPUT="$ISAAC_SIM_PATH/palletjack_sdg/palletjack_data/test_${TIMESTAMP}"

echo "Starting experiments — output root: $BASE_OUTPUT"

cd $ISAAC_SIM_PATH

for EXP_CONFIG in \
    "$EXPERIMENTS_DIR/exp1_overhead_clean.yaml" \
    "$EXPERIMENTS_DIR/exp2_low_forklift_pov.yaml" \
    "$EXPERIMENTS_DIR/exp3_wide_cluttered.yaml" \
    "$EXPERIMENTS_DIR/exp4_ceiling_telephoto.yaml" \
    "$EXPERIMENTS_DIR/exp5_mixed_realistic.yaml"
do
    EXP_NAME=$(basename "$EXP_CONFIG" .yaml)
    OUTPUT_DIR="$BASE_OUTPUT/$EXP_NAME"
    echo "--- Running $EXP_NAME ---"
    ./python.sh "$SCRIPT" --config "$EXP_CONFIG" --headless True --data_dir "$OUTPUT_DIR"
done

echo "All experiments complete. Data at: $BASE_OUTPUT"
