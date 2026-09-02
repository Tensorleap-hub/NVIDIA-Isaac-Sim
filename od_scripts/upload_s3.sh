#!/usr/bin/env bash
# Upload a finished arm's trained model + metrics to S3 (idempotent), plus the study-level files.
#   od_scripts/upload_s3.sh <arm> [<arm>...]     # per-arm: checkpoint_best_ema.pth, metrics.csv, eval/*.json, train log
#   od_scripts/upload_s3.sh --study-files        # RESULTS.md, MANIFEST.json, gt_report.html
set -uo pipefail
DC=/home/ubuntu/datasets_coco
S3=s3://nvidia-isaac-bucket/training/arms_study_20260830
if [ "${1:-}" = "--study-files" ]; then
  for f in RESULTS.md MANIFEST.json gt_report.html summary_report.html report_data.json; do
    aws s3 cp "$DC/$f" "$S3/$f" --only-show-errors && echo "up $f"
  done
  exit 0
fi
for ARM in "$@"; do
  OUT=$DC/$ARM/output/rfdetr_reducelr
  if [ ! -f "$OUT/checkpoint_best_ema.pth" ]; then echo "[$ARM] no checkpoint, skip"; continue; fi
  aws s3 cp "$OUT/checkpoint_best_ema.pth" "$S3/$ARM/checkpoint_best_ema.pth" --only-show-errors \
    && aws s3 cp "$OUT/metrics.csv" "$S3/$ARM/metrics.csv" --only-show-errors \
    && { [ -d "$OUT/eval" ] && aws s3 sync "$OUT/eval" "$S3/$ARM/eval" --only-show-errors; true; } \
    && { [ -f "$DC/logs/$ARM.train.log" ] && aws s3 cp "$DC/logs/$ARM.train.log" "$S3/$ARM/train.log" --only-show-errors; true; } \
    && echo "[$ARM] uploaded to $S3/$ARM/" || echo "[$ARM] UPLOAD FAILED"
done
