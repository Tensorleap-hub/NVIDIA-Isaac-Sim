"""Generate one sdg YAML config per row in a next-trials CSV.

Usage:
    python generate_configs.py --csv <path-to-csv> --out-dir <output-directory>

Defaults:
    --csv     next to the script (looks for any *raw_outputs.csv nearby)
    --out-dir directory named after the CSV file (without extension), next to the CSV
"""
import argparse
import csv
import os
import yaml

TEXTURE_PREFIX = "/Isaac/Materials/Textures/Patterns/"
TEXTURE_KEYS = [f"metadata.synth_metadata_synth_texture_{i}" for i in range(1, 13)]


def collect_textures(row):
    seen, textures = set(), []
    for key in TEXTURE_KEYS:
        val = row.get(key, "").strip()
        if val and val not in seen:
            seen.add(val)
            textures.append(TEXTURE_PREFIX + val)
    return textures


def build_config(row, extends_path):
    def flt(k): return round(float(row[f"metadata.synth_metadata_synth_{k}"]), 4)
    def intt(k): return int(float(row[f"metadata.synth_metadata_synth_{k}"]))

    distractors = row["metadata.synth_metadata_synth_distractors"].strip()

    return {
        "extends": extends_path,
        "run": {
            "num_frames": int(row["n_samples"]),
            "distractors": distractors,
            "data_dir": "/placeholder",   # overridden by run_experiments.sh
        },
        "camera": {
            "camera_height_min": flt("camera_height_min"),
            "camera_height_max": flt("camera_height_max"),
            "fov_min": intt("fov_min"),
            "fov_max": intt("fov_max"),
            "noise_std_min": intt("noise_std_min"),
            "noise_std_max": intt("noise_std_max"),
            "motion_blur_strength_min": flt("motion_blur_min"),
            "motion_blur_strength_max": flt("motion_blur_max"),
        },
        "lighting": {
            "intensity_mean": flt("lighting_intensity_mean"),
            "intensity_std":  flt("lighting_intensity_std"),
        },
        "materials": {
            "roughness_min": flt("materials_roughness_min"),
            "roughness_max": flt("materials_roughness_max"),
            "textures": collect_textures(row),
        },
        "palletjacks": {
            "count_per_model": intt("palletjack_count_per_model"),
        },
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Generate SDG YAML configs from a next-trials CSV.")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to the next-trials CSV file.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory where YAML configs will be written. "
                             "Created if it doesn't exist. "
                             "Defaults to a directory named after the CSV file.")
    args = parser.parse_args()

    csv_path = args.csv
    if csv_path is None:
        # Default: only CSV in the script's directory
        candidates = [f for f in os.listdir(script_dir) if f.endswith(".csv")]
        if len(candidates) == 1:
            csv_path = os.path.join(script_dir, candidates[0])
        else:
            parser.error("--csv is required when there is not exactly one CSV next to this script.")
    csv_path = os.path.abspath(csv_path)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(csv_path)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Compute relative path from out_dir back to sdg_config.yaml
    sdg_config_abs = os.path.abspath(os.path.join(script_dir, "..", "sdg_config.yaml"))
    extends_path = os.path.relpath(sdg_config_abs, out_dir)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        dist_id = row["distribution_id"].strip()
        cfg = build_config(row, extends_path)
        out_path = os.path.join(out_dir, f"{dist_id}.yaml")
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"  wrote {dist_id}.yaml")

    print(f"\nDone — {len(rows)} configs in {out_dir}")


if __name__ == "__main__":
    main()
