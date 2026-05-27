#!/bin/bash
# Usage:
#   ./run_trajectory_stage1.sh
#   ./run_trajectory_stage1.sh --num_frames 10 --headless False
#   ./run_trajectory_stage1.sh --num_frames 30 --data_dir /path/to/output

ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/opt/IsaacSim}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPT="$SCRIPT_DIR/standalone_palletjack_trajectory_sdg.py"
CONFIG="$SCRIPT_DIR/sdg_config_trajectory.yaml"
DATA_DIR="$SCRIPT_DIR/palletjack_data/trajectory_stage1"

NVJITLINK_LIB_DIR="$ISAAC_SIM_PATH/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/nvjitlink/lib"
if [ -d "$NVJITLINK_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$NVJITLINK_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "Starting trajectory stage-1 SDG"
echo "  Isaac Sim : $ISAAC_SIM_PATH"
echo "  Script    : $SCRIPT"
echo "  Config    : $CONFIG"
echo "  Data dir  : $DATA_DIR"
echo ""

cd "$ISAAC_SIM_PATH"

./python.sh "$SCRIPT" \
    --config "$CONFIG" \
    --data_dir "$DATA_DIR" \
    "$@"
