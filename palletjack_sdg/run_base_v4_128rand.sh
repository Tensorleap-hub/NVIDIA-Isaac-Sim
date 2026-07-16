#!/bin/bash
# base_v4 RANDOM-frame dataset: 128 independent snapshots per config.
# Episode mode: ONE Isaac session per config; per image the scene is fully
# re-rolled (objects, lighting, textures) and the camera gets an independent
# occupancy-validated freespace pose — no trajectory, maximally decorrelated.
# Output: <OUT_ROOT>/<exp>_seed<S>/Camera/{rgb,bounding_box_2d_tight}/ with one
# frame per seed dir (converter-compatible with the trajectory dumps).
#
# Usage:
#   ./run_base_v4_128rand.sh                    # default exp07..exp32
#   ./run_base_v4_128rand.sh exp09 exp22        # only matching configs
#   SEEDS="1 2 3" ./run_base_v4_128rand.sh      # custom seeds (default 1..128)
set -uo pipefail

ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/opt/IsaacSim}"
SEEDS="${SEEDS:-$(seq -s' ' 1 128)}"
NUM_FRAMES="${NUM_FRAMES:-1}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPT="$SCRIPT_DIR/standalone_palletjack_trajectory_sdg.py"
EXP_DIR="${EXP_DIR:-$SCRIPT_DIR/experiments/trajectory/base_v4}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/palletjack_data/trajectory/train_v4_128rand_$STAMP}"

NVJITLINK_LIB_DIR="$ISAAC_SIM_PATH/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/nvjitlink/lib"
if [ -d "$NVJITLINK_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$NVJITLINK_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# exp01-06 already have full 128-seed trajectory coverage in the 128seed dump.
DEFAULT_TOKENS="exp07 exp08 exp09 exp10 exp11 exp12 exp13 exp14 exp15 exp16 exp17 exp18 exp19 exp20 exp21 exp22 exp23 exp24 exp25 exp26 exp27 exp28 exp29 exp30 exp31 exp32"
TOKENS=("$@")
[ "${#TOKENS[@]}" -eq 0 ] && read -ra TOKENS <<< "$DEFAULT_TOKENS"

CONFIGS=()
for tok in "${TOKENS[@]}"; do
    for cfg in "$EXP_DIR"/${tok}*.yaml; do
        [ -e "$cfg" ] && CONFIGS+=("$cfg")
    done
done

N_SEEDS=$(wc -w <<< "$SEEDS")
echo "=================================================================="
echo "base_v4 random-frame dataset: ${#CONFIGS[@]} configs x $N_SEEDS seeds x $NUM_FRAMES frame(s)"
echo "  Total frames: $(( ${#CONFIGS[@]} * N_SEEDS * NUM_FRAMES ))"
echo "  Out root  : $OUT_ROOT"
echo "  Isaac Sim : $ISAAC_SIM_PATH"
echo "=================================================================="
mkdir -p "$OUT_ROOT"

cd "$ISAAC_SIM_PATH"

i=0
FAILED=()
for cfg in "${CONFIGS[@]}"; do
    base="$(basename "$cfg" .yaml)"
    i=$(( i + 1 ))
    echo ""
    echo "------------------------------------------------------------------"
    echo ">>> [$i/${#CONFIGS[@]}] $base  ($N_SEEDS random-frame episodes, one session)"
    echo "------------------------------------------------------------------"
    if ./python.sh "$SCRIPT" --config "$cfg" --num_frames "$NUM_FRAMES" \
            --capture_mode random --seeds "$SEEDS" --out_root "$OUT_ROOT" \
            --headless True; then
        echo "<<< $base OK"
    else
        # Nonzero exit = some episodes exhausted in-process retries (or the
        # session crashed). Per-episode dirs that DID finish are kept.
        echo "<<< $base HAD FAILURES (see log above)"
        FAILED+=("$base")
    fi
done

echo ""
echo "=================================================================="
echo "Output: $OUT_ROOT"
if [ "${#FAILED[@]}" -eq 0 ]; then
    echo "All ${#CONFIGS[@]} configs completed."
else
    echo "${#FAILED[@]}/${#CONFIGS[@]} configs had failures: ${FAILED[*]}"
    exit 1
fi
