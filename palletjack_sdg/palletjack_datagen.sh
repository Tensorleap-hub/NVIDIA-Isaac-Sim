#!/bin/bash

# This is the path where Isaac Sim is installed which contains the python.sh script
ISAAC_SIM_PATH='/isaac-sim'
SCRIPT="$ISAAC_SIM_PATH/palletjack_sdg/standalone_palletjack_sdg.py"
CONFIG="$ISAAC_SIM_PATH/palletjack_sdg/sdg_config.yaml"
BASE_DATA_DIR="$ISAAC_SIM_PATH/palletjack_sdg/palletjack_data"

echo "Starting Data Generation"

cd $ISAAC_SIM_PATH

./python.sh $SCRIPT \
    --config $CONFIG \
    --headless True \
    --distractors warehouse \
    --num_frames 2000 \
    --data_dir $BASE_DATA_DIR/distractors_warehouse

./python.sh $SCRIPT \
    --config $CONFIG \
    --headless True \
    --distractors additional \
    --num_frames 2000 \
    --data_dir $BASE_DATA_DIR/distractors_additional

./python.sh $SCRIPT \
    --config $CONFIG \
    --headless True \
    --distractors None \
    --num_frames 1000 \
    --data_dir $BASE_DATA_DIR/no_distractors
