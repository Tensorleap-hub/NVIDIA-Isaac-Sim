#!/bin/bash

# This is the path where Isaac Sim is installed which contains the python.sh script
ISAAC_SIM_PATH="/home/ubuntu/IsaacSim"

## Go to location of the SDG script
cd ../palletjack_sdg
SCRIPT_PATH="${PWD}/standalone_palletjack_sdg.py"
CONFIG_PATH="${PWD}/sdg_config.yaml"
OUTPUT_WAREHOUSE="${PWD}/palletjack_data/distractors_warehouse"
OUTPUT_ADDITIONAL="${PWD}/palletjack_data/distractors_additional"
OUTPUT_NO_DISTRACTORS="${PWD}/palletjack_data/no_distractors"


## Go to Isaac Sim location for running with ./python.sh
cd $ISAAC_SIM_PATH

echo "Starting Data Generation"

./python.sh $SCRIPT_PATH --config $CONFIG_PATH --height 544 --width 960 --num_frames 20 --distractors warehouse --data_dir $OUTPUT_WAREHOUSE

./python.sh $SCRIPT_PATH --config $CONFIG_PATH --height 544 --width 960 --num_frames 20 --distractors additional --data_dir $OUTPUT_ADDITIONAL

./python.sh $SCRIPT_PATH --config $CONFIG_PATH --height 544 --width 960 --num_frames 20 --distractors None --data_dir $OUTPUT_NO_DISTRACTORS


