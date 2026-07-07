#!/bin/bash
# Build the base_v4 "tight-framing / de-correlated" trajectory dataset.
# Each run: 10 frames, LOW frame rate (capture_dt=5.0 => 0.2 fps) so the
# arc-length-spaced frames land far apart, over a long planned path.
# Output: <OUT_ROOT>/<exp>_seed<S>/  (matches train_v1/train_v2 naming).
#
#   21 configs x 13 seeds x 10 frames = 1300 frames.
#
# Usage:
#   ./run_base_v4_train.sh                        # all 21 configs x 13 seeds
#   ./run_base_v4_train.sh exp02 exp09            # only matching configs
#   SEEDS="42 123" ./run_base_v4_train.sh         # custom seed set
#   NUM_FRAMES=2 ./run_base_v4_train.sh exp01     # smoke
#   OUT_ROOT=/path ./run_base_v4_train.sh         # custom output root
set -uo pipefail

ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/opt/IsaacSim}"
NUM_FRAMES="${NUM_FRAMES:-10}"
# capture_dt (fps) is NOT a frame-spacing knob for camera_rig configs: ego frames
# are interpolated by arc-length over the planned path, so spacing =
# path_length / (num_frames-1), independent of fps. We therefore do NOT force a
# capture_dt here — each config uses its own value. Set the CAPTURE_DT env var
# only to override deliberately (e.g. for people-animation spread).
CAPTURE_DT="${CAPTURE_DT:-}"
# 13 seeds -> 13 independent scene layouts + start points per config.
SEEDS="${SEEDS:-42 123 456 789 101 202 303 404 505 606 707 808 909}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPT="$SCRIPT_DIR/standalone_palletjack_trajectory_sdg.py"
EXP_DIR="$SCRIPT_DIR/experiments/trajectory/base_v4"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/palletjack_data/trajectory/train_v4_$STAMP}"

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
echo "base_v4 dataset: ${#CONFIGS[@]} configs x ${#SEED_ARR[@]} seeds = $TOTAL runs"
echo "  Frames/run: $NUM_FRAMES  (capture_dt: ${CAPTURE_DT:-per-config default})"
echo "  Total frames: $(( TOTAL * NUM_FRAMES ))"
echo "  Seeds     : ${SEEDS}"
echo "  Out root  : $OUT_ROOT"
echo "  Isaac Sim : $ISAAC_SIM_PATH"
echo "=================================================================="
mkdir -p "$OUT_ROOT"

# Only pass --capture_dt when explicitly overridden; otherwise each config decides.
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
        echo "------------------------------------------------------------------"
        echo ">>> [$i/$TOTAL] $name"
        echo "------------------------------------------------------------------"
        # Seed-retry: a run fails when THIS random scene layout leaves no
        # navigable freespace (empty buffered map -> UniformPoseSampler
        # "high <= 0", or no >=min_path_m route -> "No valid occupancy path").
        # That's a property of the layout, not the config, so resampling with a
        # fresh seed (seed + k*1000) almost always recovers it — WITHOUT relaxing
        # the strict occupancy buffer (which would risk wall/shelf clipping). The
        # output dir keeps the original seed label so we still get one dir per
        # intended slot. MAX_SEED_RETRIES=0 restores the old fail-fast behavior.
        max_retries="${MAX_SEED_RETRIES:-4}"
        ok=0
        for attempt in $(seq 0 "$max_retries"); do
            try_seed=$(( seed + attempt * 1000 ))
            [ "$attempt" -gt 0 ] && echo "    retry $attempt/$max_retries with seed $try_seed"
            if ./python.sh "$SCRIPT" --config "$cfg" --num_frames "$NUM_FRAMES" \
                    "${CAPTURE_ARG[@]}" --seed "$try_seed" \
                    --data_dir "$OUT_ROOT/$name" --headless True; then
                ok=1; break
            fi
        done
        if [ "$ok" -eq 1 ]; then
            echo "<<< $name OK${try_seed:+ (seed $try_seed)}"
        else
            echo "<<< $name FAILED after $((max_retries+1)) attempts"
            FAILED+=("$name")
        fi
    done
done

echo ""
echo "=================================================================="
echo "Output: $OUT_ROOT"
if [ "${#FAILED[@]}" -eq 0 ]; then
    echo "All $TOTAL runs completed."
else
    echo "${#FAILED[@]}/$TOTAL failed: ${FAILED[*]}"
    exit 1
fi
