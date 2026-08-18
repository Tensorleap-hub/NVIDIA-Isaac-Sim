#!/usr/bin/env bash
# Sync the local data needed to reproduce the cosmos Tensorleap push.
#
# For each required folder/file below:
#   - if it already exists under the S3 prefix -> download it (aws s3 sync, local becomes a copy of S3)
#   - if it is missing from S3 but exists locally -> upload it (populate S3 for the next person)
#   - if it is missing from both -> report it and keep going
#
# S3 prefix: s3://nvidia-isaac-bucket/cosmos-presentation-data/
#
# Usage:
#   ./scripts/sync_cosmos_presentation_data.sh [--dry-run] [--local-root DIR] [--s3-prefix S3_URI]
#
# --local-root defaults to the warehouse root configured in
# tensorleap_intgration_code/project_config.yaml (data.data_path). Pass --local-root
# to point at a different machine's copy of the warehouse.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$REPO_ROOT/tensorleap_intgration_code/project_config.yaml"

S3_PREFIX="s3://nvidia-isaac-bucket/cosmos-presentation-data"
LOCAL_ROOT="$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['data']['data_path'])")"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --local-root) LOCAL_ROOT="$2"; shift 2 ;;
    --s3-prefix) S3_PREFIX="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done
S3_PREFIX="${S3_PREFIX%/}"

echo "Local warehouse root: $LOCAL_ROOT"
echo "S3 prefix:            $S3_PREFIX"
[[ "$DRY_RUN" == 1 ]] && echo "(dry run: no data will move)"
echo

# --- entries synced as full directories, relative to $LOCAL_ROOT --------------
WAREHOUSE_DIRS=(
  "dataset/labels"
  "dataset/subset-2"
  "dataset/subset-3"
  "base_v2_final"
  "warehouse3cls_cosmos_themes"
  "warehouse3cls_cosmos_themes_original"
  "warehouse3cls_cosmos_optuna"
  "warehouse3cls_cosmos_optuna_original"
)

# --- entries where only run_config.yaml files matter (the raw renders are huge
# and are not read by the integration, only the per-run config metadata is) ----
RUN_CONFIG_ONLY_DIRS=(
  "20260708_cosmos_v4"
  "20260712_cosmos_v4"
  "20260712_cosmos_optuna"
)

# --- small repo-root CSVs used by sample_selection_filter, synced individually -
REPO_FILES=(
  "comparison_subset3_proximity_by_ls.csv"
  "comparison_subset3_proximity_by_ls_base_synth_bbox80.csv"
  "manual_base_tlopt_selection_500.csv"
)

run() {
  echo "+ $*"
  [[ "$DRY_RUN" == 1 ]] || "$@"
}

s3_has_content() {
  aws s3 ls "$1/" >/dev/null 2>&1
}

sync_dir() {
  local rel="$1"; shift
  local extra_args=("$@")
  local local_dir="$LOCAL_ROOT/$rel"
  local s3_dir="$S3_PREFIX/$rel"

  if s3_has_content "$s3_dir"; then
    echo "[download] $rel  (S3 -> local)"
    mkdir -p "$local_dir"
    run aws s3 sync "$s3_dir/" "$local_dir/" "${extra_args[@]+"${extra_args[@]}"}"
  elif [[ -d "$local_dir" ]]; then
    echo "[upload]   $rel  (local -> S3, populating cosmos-presentation-data)"
    run aws s3 sync "$local_dir/" "$s3_dir/" "${extra_args[@]+"${extra_args[@]}"}"
  else
    echo "[MISSING]  $rel  -- not found in S3 or locally, skipping"
  fi
}

sync_file() {
  local rel="$1"
  local local_file="$REPO_ROOT/$rel"
  local s3_file="$S3_PREFIX/repo-files/$rel"

  if aws s3api head-object --bucket "$(echo "$S3_PREFIX" | sed -E 's#s3://([^/]+)/.*#\1#')" \
      --key "$(echo "$S3_PREFIX/repo-files/$rel" | sed -E 's#s3://[^/]+/##')" >/dev/null 2>&1; then
    echo "[download] $rel  (S3 -> repo root)"
    run aws s3 cp "$s3_file" "$local_file"
  elif [[ -f "$local_file" ]]; then
    echo "[upload]   $rel  (repo root -> S3)"
    run aws s3 cp "$local_file" "$s3_file"
  else
    echo "[MISSING]  $rel  -- not found in S3 or locally, skipping"
  fi
}

echo "== warehouse directories =="
for rel in "${WAREHOUSE_DIRS[@]}"; do
  sync_dir "$rel"
done

echo
echo "== run_config.yaml-only directories =="
for rel in "${RUN_CONFIG_ONLY_DIRS[@]}"; do
  sync_dir "$rel" --exclude "*" --include "*/run_config.yaml" --include "run_config.yaml"
done

echo
echo "== repo-root CSVs (sample_selection_filter) =="
for rel in "${REPO_FILES[@]}"; do
  sync_file "$rel"
done

echo
echo "Done."
