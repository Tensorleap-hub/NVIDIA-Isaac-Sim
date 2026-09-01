#!/usr/bin/env bash
# A resumed Lightning run's CSVLogger starts a fresh metrics.csv (loses pre-resume history on
# disk, even though the run itself correctly continues from the checkpoint's epoch/LR/EMA state).
# This splices a saved pre-resume metrics.csv back in front of the current one, keeping only ONE
# header line. Run once, after the resumed run has finished (or at least logged its first epoch).
#   od_scripts/merge_metrics.sh <pre_resume.csv> <output_dir>/metrics.csv
set -euo pipefail
PRE=$1; CUR=$2
[ -f "$CUR.premerge_backup" ] && { echo "already merged (backup exists), skip"; exit 0; }
cp "$CUR" "$CUR.premerge_backup"
{ head -1 "$PRE"; tail -n +2 "$PRE"; tail -n +2 "$CUR"; } > "$CUR.tmp"
mv "$CUR.tmp" "$CUR"
echo "merged $(wc -l < "$PRE") pre-resume + $(($(wc -l < "$CUR.premerge_backup")-0)) resumed rows -> $(wc -l < "$CUR") total, backup at $CUR.premerge_backup"
