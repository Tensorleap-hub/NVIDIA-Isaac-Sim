"""
Generate synth images for a TL next-trials CSV and append to the synth metadata.

Usage:
    python -m convergence.generate_tl_iter <path-to-next-trials.csv> [--iter N]

Reads the CSV produced by TL (columns: distribution_id, simulation_type, n_samples,
metadata.theta_*), generates images for each distribution, and saves them under:

    <DATA_ROOT>/tl_iter_<N>/images/<dist_id>_<i>.png

A per-iteration metadata.csv is written at:

    <DATA_ROOT>/tl_iter_<N>/metadata.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_ROOT, THETA_KEYS, N_IMAGES_PER_TRIAL
from .generator import generate_images

_THETA_PREFIX = "metadata.theta_"


def _parse_theta(row: pd.Series) -> dict:
    theta = {}
    for k in THETA_KEYS:
        col = f"{_THETA_PREFIX}{k}"
        val = float(row[col])
        if k == "clutter_count":
            val = int(round(val))
        theta[k] = val
    return theta


def main(csv_path: str, iter_n: int) -> None:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    out_dir = DATA_ROOT / f"tl_iter_{iter_n}"
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    for row_i, (_, row) in enumerate(df.iterrows()):
        dist_id = str(row["distribution_id"])
        n = int(row.get("n_samples", N_IMAGES_PER_TRIAL))
        theta = _parse_theta(row)
        images = generate_images(theta, n=n, seed=row_i)
        for img_i, img in enumerate(images):
            fname = f"{dist_id}_{img_i:04d}.png"
            img.save(images_dir / fname)
            metadata_rows.append({"image_path": str(images_dir / fname), **theta})
        if (row_i + 1) % 5 == 0:
            print(f"  {row_i + 1}/{len(df)} distributions done")

    meta_df = pd.DataFrame(metadata_rows, columns=["image_path"] + THETA_KEYS)
    meta_df.to_csv(out_dir / "metadata.csv", index=False)
    print(f"Saved {len(meta_df)} rows → {out_dir / 'metadata.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to TL next-trials CSV")
    parser.add_argument("--iter", type=int, default=1, dest="iter_n", help="Iteration number (default: 1)")
    args = parser.parse_args()
    main(args.csv_path, args.iter_n)
