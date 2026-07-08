#!/bin/bash
# Generate the cosmos_v4 dataset — dense-in-time VIDEO clips for Cosmos
# augmentation (optimization-recipe.md, Pipeline A / "Two temporal densities").
# Each run: 128 frames @ 30 fps (capture_dt from config) + MP4 (capture.video=true).
# Style is identical to base_v4 (the OD set); only timing + video differ.
#
# NOTE: video clips are heavy. Default seed set is small (each seed = one full
# 128-frame clip). capture_dt is NOT overridden here so each config's 30 fps holds.
#
# Usage:
#   ./run_cosmos_v4_train.sh                      # all present configs x default seeds
#   ./run_cosmos_v4_train.sh exp01 exp06          # subset
#   SEEDS="42" NUM_FRAMES=16 ./run_cosmos_v4_train.sh exp01   # smoke
set -uo pipefail

ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/opt/IsaacSim}"
NUM_FRAMES="${NUM_FRAMES:-128}"          # dense clip
CAPTURE_DT="${CAPTURE_DT:-}"             # empty -> use config's 30 fps (0.0333)
SEEDS="${SEEDS:-42 123 456 789}"         # fewer episodes: each seed is a full clip

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPT="$SCRIPT_DIR/standalone_palletjack_trajectory_sdg.py"
EXP_DIR="$SCRIPT_DIR/experiments/trajectory/cosmos_v4"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/palletjack_data/trajectory/cosmos_v4_$STAMP}"

NVJITLINK_LIB_DIR="$ISAAC_SIM_PATH/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/nvjitlink/lib"
if [ -d "$NVJITLINK_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$NVJITLINK_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mapfile -t CONFIGS < <(ls "$EXP_DIR"/exp*.yaml | sort)
if [ "$#" -gt 0 ]; then
    FILTERED=()
    for cfg in "${CONFIGS[@]}"; do
        for tok in "$@"; do
            [[ "$(basename "$cfg")" == *"$tok"* ]] && FILTERED+=("$cfg") && break
        done
    done
    CONFIGS=("${FILTERED[@]}")
fi

read -ra SEED_ARR <<< "$SEEDS"
TOTAL=$(( ${#CONFIGS[@]} * ${#SEED_ARR[@]} ))
echo "=================================================================="
echo "cosmos_v4 (video): ${#CONFIGS[@]} configs x ${#SEED_ARR[@]} seeds = $TOTAL clips"
echo "  Frames/clip: $NUM_FRAMES  (capture_dt: ${CAPTURE_DT:-per-config 30fps}) + MP4"
echo "  Out root  : $OUT_ROOT"
echo "=================================================================="
mkdir -p "$OUT_ROOT"

CAPTURE_ARG=()
[ -n "$CAPTURE_DT" ] && CAPTURE_ARG=(--capture_dt "$CAPTURE_DT")

cd "$ISAAC_SIM_PATH"

i=0
FAILED=()
for cfg in "${CONFIGS[@]}"; do
    base="$(basename "$cfg" .yaml)"
    for seed in "${SEED_ARR[@]}"; do
        i=$(( i + 1 ))
        name="${base}_seed${seed}"
        echo ""
        echo ">>> [$i/$TOTAL] $name"
        if ./python.sh "$SCRIPT" --config "$cfg" --num_frames "$NUM_FRAMES" \
                "${CAPTURE_ARG[@]}" --seed "$seed" \
                --data_dir "$OUT_ROOT/$name" --headless True; then
            echo "<<< $name OK"
        else
            echo "<<< $name FAILED (rc=$?)"
            FAILED+=("$name")
        fi
    done
done

echo ""
echo "Output: $OUT_ROOT"
if [ "${#FAILED[@]}" -eq 0 ]; then
    echo "All $TOTAL clips completed."
else
    echo "${#FAILED[@]}/$TOTAL failed: ${FAILED[*]}"
    exit 1
fi
