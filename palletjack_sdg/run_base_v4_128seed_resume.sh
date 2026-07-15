#!/bin/bash
# Resume of the 128-seed base_v4 run (train_v4_128seed_20260713_174507) after
# restructuring to reuse the v4b dump (train_v4_20260707_181129: exp01-21 x 20
# seeds, rendered post wall-fix + bounds-fix, so domain-equivalent).
#
#   exp01-06 : already complete at 128 seeds (plus exp01 backfill of 15-18 here)
#   exp07-21 : 108 new seeds (1..112 minus v4b's 42/101/111/123) + 20 v4b = 128 layouts
#   exp22-32 : full 128 seeds (no prior coverage)
set -uo pipefail
cd "$(dirname "$0")"

OUT_ROOT=/home/ubuntu/NVIDIA-Isaac-Sim/palletjack_sdg/palletjack_data/trajectory/train_v4_128seed_20260713_174507
export OUT_ROOT NUM_FRAMES=5

S108=$(for i in $(seq 1 112); do case $i in 42|101|111|123) ;; *) printf '%s ' "$i";; esac; done)
S128=$(seq -s' ' 1 128)

echo "### stage 0: exp01 backfill (seeds 15-18)"
SEEDS="15 16 17 18" ./run_base_v4_train.sh exp01

echo "### stage 1: exp07-exp21 x 108 seeds"
SEEDS="$S108" ./run_base_v4_train.sh exp07 exp08 exp09 exp10 exp11 exp12 exp13 exp14 exp15 exp16 exp17 exp18 exp19 exp20 exp21

echo "### stage 2: exp22-exp32 x 128 seeds"
SEEDS="$S128" ./run_base_v4_train.sh exp22 exp23 exp24 exp25 exp26 exp27 exp28 exp29 exp30 exp31 exp32

echo "### resume wrapper done"
