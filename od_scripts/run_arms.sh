#!/usr/bin/env bash
# Sequential training queue for the 4-arm study. One launch:
#   nohup od_scripts/run_arms.sh > /home/ubuntu/datasets_coco/logs/queue.log 2>&1 &
# Per arm: train (ReduceLROnPlateau recipe) -> fair evals of checkpoint_best_ema.pth on
#   * real/valid (LOCO subset-3)          <- the selection metric
#   * evalsets/train_real                 <- fit on the real training images
#   * evalsets/train_<synth> per source   <- fit on each synthetic source (diagnostic only)
#   * the arm's own train/ (combined)     <- fit on everything it trained on
# Arms can be given as args (default: every arm in common.ARMS, in order).
set -uo pipefail
REPO=/home/ubuntu/NVIDIA-Isaac-Sim
PY=$REPO/.venv/bin/python
OD=$REPO/od_scripts
DC=/home/ubuntu/datasets_coco
LOGS=$DC/logs
mkdir -p "$LOGS"
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=($("$PY" -c "import sys; sys.path.insert(0,'$OD'); from common import ARMS; print(' '.join(ARMS))"))

sources_of() {  # synthetic sources per arm, from common.ARMS
  "$PY" -c "import sys; sys.path.insert(0,'$OD'); from common import ARMS; print(' '.join(ARMS['$1']))"
}

# Env knobs: EPOCHS_DEFAULT (cap, default 60), EPOCHS_<arm> (per-arm cap), NUM_WORKERS (default 4),
#            REUSE_EXISTING=1 -> if <arm>/output/rfdetr_reducelr/checkpoint_best_ema.pth exists, skip training (evals only).
for ARM in "${ARMS[@]}"; do
  OUT=$DC/$ARM/output/rfdetr_reducelr
  CKPT=$OUT/checkpoint_best_ema.pth
  EPV="EPOCHS_$ARM"; EPOCHS=${!EPV:-${EPOCHS_DEFAULT:-60}}
  if [ "${REUSE_EXISTING:-0}" = "1" ] && [ -f "$CKPT" ]; then
    echo "$(date -u +%FT%TZ) [$ARM] reusing existing $CKPT (training skipped)"; RC=0
  else
    echo "$(date -u +%FT%TZ) [$ARM] train start (epochs cap $EPOCHS, workers ${NUM_WORKERS:-4})"
    rm -rf "$OUT"                                 # never let rf-detr silently resume from last.ckpt
    "$PY" "$OD/train.py" --dataset-dir "$DC/$ARM" --output-dir "$OUT" --epochs "$EPOCHS" \
          --num-workers "${NUM_WORKERS:-4}" > "$LOGS/$ARM.train.log" 2>&1
    RC=$?
    echo "$(date -u +%FT%TZ) [$ARM] train rc=$RC"
  fi
  if [ $RC -ne 0 ] || [ ! -f "$CKPT" ]; then echo "[$ARM] FAILED (no $CKPT)"; continue; fi

  EV=$OUT/eval; mkdir -p "$EV"
  declare -A SPLITS=( [valid_real]="$DC/real" [train_real]="$DC/evalsets/train_real" [train_combined]="$DC/evalsets/train_$ARM" )
  for S in $(sources_of "$ARM"); do SPLITS[train_$S]="$DC/evalsets/train_$S"; done
  # combined train of this arm as an eval set (valid -> its own train/)
  if [ ! -d "$DC/evalsets/train_$ARM" ]; then
    mkdir -p "$DC/evalsets/train_$ARM"
    ln -s "../../$ARM/train" "$DC/evalsets/train_$ARM/valid"
    ln -s "../../$ARM/train" "$DC/evalsets/train_$ARM/train"
  fi
  for NAME in "${!SPLITS[@]}"; do
    echo "$(date -u +%FT%TZ) [$ARM] eval $NAME"
    "$PY" "$OD/eval_checkpoint.py" --dataset-dir "${SPLITS[$NAME]}" --pretrain-weights "$CKPT" \
         --json "$EV/$NAME.json" --output-dir "$LOGS/eval_tmp" > "$LOGS/$ARM.eval_$NAME.log" 2>&1 \
      || echo "[$ARM] eval $NAME FAILED"
  done
  unset SPLITS
  "$PY" "$OD/results.py" > "$DC/RESULTS.md" 2>/dev/null && echo "$(date -u +%FT%TZ) [$ARM] results table updated"
done
echo "$(date -u +%FT%TZ) queue complete"
