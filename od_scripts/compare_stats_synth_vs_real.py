"""
Statistical comparison across multiple synth/real data sources.

Computes per-box statistics (normalized by image size):
  - class distribution
  - relative box area  (w*h / img_w*img_h)
  - relative width  (w / img_w)
  - relative height (h / img_h)
  - aspect ratio    (w / h)
  - center position (cx/img_w, cy/img_h)
  - boxes per image

Supported input formats:
  --synth-root   selected_trial_downloads/optuna-ec2/  (best_* Isaac Sim npy+json)
  --base-dir     palletjack_run_0/                     (KITTI-format object_detection txts)
  --loco-ann     loco_dataset/labels/loco-*.json       (COCO JSON)

Outputs a multi-panel matplotlib figure and a printed summary table.

Usage:
    python scripts/compare_stats_synth_vs_real.py \\
        --synth-root /path/to/selected_trial_downloads/optuna-ec2 \\
        --base-dir   /path/to/palletjack_run_0 \\
        --loco-ann   /path/to/loco_dataset/labels/loco-sub3-v1-train.json \\
        --output     output/stats_all_sources.png
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

KEEP_CLASSES    = {"pallet_truck", "forklift", "pallet"}
SYNTH_CLASS_MAP = {"palletjack": "pallet_truck", "forklift": "forklift", "pallet": "pallet"}
CLASS_ORDER     = ["pallet_truck", "forklift", "pallet"]

# Color per source label
SOURCE_PALETTE = [
    "#4C9BE8",   # blue   – synth best trials
    "#E8C84C",   # yellow – base run
    "#E87C4C",   # orange – real LOCO
    "#7CE87C",   # green  – any extra source
]


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _box_record(cls, x0, y0, x1, y1, img_w, img_h, source, **extra) -> dict:
    w = x1 - x0
    h = y1 - y0
    return {
        "class":        cls,
        "rel_w":        w / img_w,
        "rel_h":        h / img_h,
        "rel_area":     (w * h) / (img_w * img_h),
        "aspect_ratio": w / h,
        "cx_rel":       (x0 + w / 2) / img_w,
        "cy_rel":       (y0 + h / 2) / img_h,
        "img_w":        img_w,
        "img_h":        img_h,
        "source":       source,
        **extra,
    }


def _read_run_dir(run_dir: Path, source_label: str, theme: str) -> list[dict]:
    """Read one Isaac Sim outputs/<run_dir> and return box records."""
    from PIL import Image as PILImage

    rgb_files   = {f.stem.split("_")[-1]: f for f in run_dir.glob("rgb_*.png")}
    bbox_files  = {f.stem.split("_")[-1]: f for f in run_dir.glob("bounding_box_2d_tight_[0-9]*.npy")}
    label_files = {f.stem.split("_")[-1]: f for f in run_dir.glob("bounding_box_2d_tight_labels_[0-9]*.json")}
    complete = sorted(set(rgb_files) & set(bbox_files) & set(label_files))

    records = []
    for n in complete:
        bboxes = np.load(bbox_files[n], allow_pickle=True)
        with open(label_files[n]) as f:
            label_map = json.load(f)

        sem_to_class = {
            int(k): SYNTH_CLASS_MAP[v["class"]]
            for k, v in label_map.items()
            if SYNTH_CLASS_MAP.get(v.get("class", "")) in KEEP_CLASSES
        }
        if not sem_to_class:
            continue

        with PILImage.open(rgb_files[n]) as img:
            img_w, img_h = img.size

        for row in bboxes:
            sem_id = int(row["semanticId"])
            if sem_id not in sem_to_class:
                continue
            x0, y0 = int(row["x_min"]), int(row["y_min"])
            x1, y1 = int(row["x_max"]), int(row["y_max"])
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                continue
            records.append(_box_record(
                sem_to_class[sem_id], x0, y0, x1, y1, img_w, img_h,
                source_label,
                occlusionRatio=float(row["occlusionRatio"]),
                theme=theme,
            ))
    return records


def collect_synth_stats(synth_root: Path, source_label: str = "synth") -> list[dict]:
    """
    Walk Isaac Sim outputs in any layout by finding all 'outputs/<run_dir>'
    subtrees under synth_root.  Follows symlinks. Handles:
      - root/theme/cycle/best_*/outputs/run_dir/
      - root/workspace/trial_N/outputs/run_dir/
      - root/<symlink_to_trial>/outputs/run_dir/
    """
    import os
    records = []
    for dirpath, dirnames, _ in os.walk(synth_root, followlinks=True):
        dirnames.sort()
        if Path(dirpath).name == "outputs":
            theme = Path(dirpath).parent.name
            for run_name in sorted(dirnames):
                run_dir = Path(dirpath) / run_name
                records.extend(_read_run_dir(run_dir, source_label, theme))
            dirnames.clear()  # don't recurse further into outputs
    return records


def collect_base_stats(base_dir: Path, source_label: str = "base",
                       exp_filter: list[str] | None = None) -> list[dict]:
    """
    Walk base_dir/<exp>/ in Isaac Sim npy+json format:
        rgb_XXXX.png
        bounding_box_2d_tight_XXXX.npy
        bounding_box_2d_tight_labels_XXXX.json

    exp_filter: if given, only include experiments whose name contains any of
                these strings (case-insensitive).
    """
    from PIL import Image as PILImage

    records = []
    for exp_dir in sorted(base_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_filter and not any(f.lower() in exp_dir.name.lower() for f in exp_filter):
            continue

        rgb_files   = {f.stem.split("_")[-1]: f for f in exp_dir.glob("rgb_*.png")}
        bbox_files  = {f.stem.split("_")[-1]: f for f in exp_dir.glob("bounding_box_2d_tight_[0-9]*.npy")}
        label_files = {f.stem.split("_")[-1]: f for f in exp_dir.glob("bounding_box_2d_tight_labels_[0-9]*.json")}
        complete = sorted(set(rgb_files) & set(bbox_files) & set(label_files))

        for n in complete:
            bboxes = np.load(bbox_files[n], allow_pickle=True)
            with open(label_files[n]) as f:
                label_map = json.load(f)

            sem_to_class = {
                int(k): SYNTH_CLASS_MAP[v["class"]]
                for k, v in label_map.items()
                if SYNTH_CLASS_MAP.get(v.get("class", "")) in KEEP_CLASSES
            }
            if not sem_to_class:
                continue

            with PILImage.open(rgb_files[n]) as img:
                img_w, img_h = img.size

            for row in bboxes:
                sem_id = int(row["semanticId"])
                if sem_id not in sem_to_class:
                    continue
                x0, y0 = int(row["x_min"]), int(row["y_min"])
                x1, y1 = int(row["x_max"]), int(row["y_max"])
                if x1 - x0 <= 0 or y1 - y0 <= 0:
                    continue
                records.append(_box_record(
                    sem_to_class[sem_id], x0, y0, x1, y1, img_w, img_h,
                    source_label,
                    occlusionRatio=float(row["occlusionRatio"]),
                    exp=exp_dir.name,
                ))
    return records


def collect_loco_stats(ann_path: Path, source_label: str = "real") -> list[dict]:
    """Load LOCO COCO annotation JSON."""
    with open(ann_path) as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_meta = {img["id"]: img for img in coco["images"]}

    records = []
    for ann in coco["annotations"]:
        cls = cat_id_to_name.get(ann["category_id"])
        if cls not in KEEP_CLASSES:
            continue
        img = img_meta.get(ann["image_id"])
        if img is None:
            continue
        img_w, img_h = img["width"], img["height"]
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        records.append(_box_record(
            cls, x, y, x + w, y + h, img_w, img_h,
            source_label,
        ))
    return records


# ---------------------------------------------------------------------------
# Image-level statistics
# ---------------------------------------------------------------------------

def _box_blur(a: np.ndarray, k: int) -> np.ndarray:
    H, W = a.shape
    padded = np.pad(a, k // 2, mode='reflect')        # shape (H+k-1, W+k-1) for odd k
    cs = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    cs[1:, 1:] = padded.cumsum(0).cumsum(1)
    return (cs[k:k+H, k:k+W] - cs[0:H, k:k+W] - cs[k:k+H, 0:W] + cs[0:H, 0:W]) / (k * k)


def compute_image_stats(img: np.ndarray) -> dict:
    """H×W×3 uint8 → dict of scalar image statistics."""
    rgb = img.astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    sat = np.where(cmax > 0, (cmax - cmin) / cmax, 0.0)

    lap = (
        -4 * luma[1:-1, 1:-1]
        + luma[:-2, 1:-1] + luma[2:, 1:-1]
        + luma[1:-1, :-2] + luma[1:-1, 2:]
    )

    hp = luma - _box_blur(luma, 15)
    hp_rms = float(np.sqrt((hp ** 2).mean()))
    luma_rms = float(np.sqrt((luma ** 2).mean()))

    noise = luma - _box_blur(luma, 3)

    gf = luma * 255.0
    sx = (gf[:-2, 2:] - gf[:-2, :-2] + 2*gf[1:-1, 2:] - 2*gf[1:-1, :-2] + gf[2:, 2:] - gf[2:, :-2]) / 8.0
    sy = (gf[2:, :-2] - gf[:-2, :-2] + 2*gf[2:, 1:-1] - 2*gf[:-2, 1:-1] + gf[2:, 2:] - gf[:-2, 2:]) / 8.0

    bp5, bp50, bp95 = np.percentile(luma, [5, 50, 95])

    return {
        "mean_r": float(r.mean()), "mean_g": float(g.mean()), "mean_b": float(b.mean()),
        "std_r":  float(r.std()),  "std_g":  float(g.std()),  "std_b":  float(b.std()),
        "bright_p5": float(bp5), "bright_p50": float(bp50), "bright_p95": float(bp95),
        "sat_mean": float(sat.mean()), "sat_std": float(sat.std()),
        "contrast":       float(luma.std()),
        "laplacian_var":  float(lap.var()),
        "highpass_ratio": hp_rms / (luma_rms + 1e-8),
        "noise_residual": float(noise.std()),
        "edge_density":   float((np.sqrt(sx**2 + sy**2) > 10.0).mean()),
    }


def collect_loco_image_stats(ann_path: Path, img_root: Path,
                             source_label: str = "real") -> list[dict]:
    """Compute image stats for every image referenced in a LOCO COCO JSON."""
    from PIL import Image as PILImage

    with open(ann_path) as f:
        coco = json.load(f)

    stats = []
    for img_meta in coco["images"]:
        rel = img_meta["path"].removeprefix("/dataset/")
        img_path = img_root / rel
        if not img_path.exists():
            continue
        arr = np.array(PILImage.open(img_path).convert("RGB"))
        s = compute_image_stats(arr)
        s["source"] = source_label
        stats.append(s)
    return stats


def collect_synth_image_stats(synth_root: Path,
                              source_label: str = "synth") -> list[dict]:
    """Compute image stats for all rgb_*.png files in an Isaac Sim outputs tree."""
    import os
    from PIL import Image as PILImage

    stats = []
    for dirpath, dirnames, _ in os.walk(synth_root, followlinks=True):
        dirnames.sort()
        if Path(dirpath).name == "outputs":
            for run_name in sorted(os.listdir(dirpath)):
                run_dir = Path(dirpath) / run_name
                if not run_dir.is_dir():
                    continue
                for png in sorted(run_dir.glob("rgb_*.png")):
                    arr = np.array(PILImage.open(png).convert("RGB"))
                    s = compute_image_stats(arr)
                    s["source"] = source_label
                    stats.append(s)
            dirnames.clear()
    return stats


def collect_base_image_stats(base_dir: Path, source_label: str = "base",
                             exp_filter: list[str] | None = None) -> list[dict]:
    """Compute image stats for rgb_*.png files in a base-run exp directory tree."""
    from PIL import Image as PILImage

    stats = []
    for exp_dir in sorted(base_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_filter and not any(f.lower() in exp_dir.name.lower() for f in exp_filter):
            continue
        for png in sorted(exp_dir.glob("rgb_*.png")):
            arr = np.array(PILImage.open(png).convert("RGB"))
            s = compute_image_stats(arr)
            s["source"] = source_label
            stats.append(s)
    return stats


IMAGE_STAT_PANELS = [
    ("mean_r",         "Mean R",                  (0.0, 1.0)),
    ("mean_g",         "Mean G",                  (0.0, 1.0)),
    ("mean_b",         "Mean B",                  (0.0, 1.0)),
    ("contrast",       "Contrast (luma std)",      (0.0, 0.5)),
    ("std_r",          "Std R",                   (0.0, 0.5)),
    ("std_g",          "Std G",                   (0.0, 0.5)),
    ("std_b",          "Std B",                   (0.0, 0.5)),
    ("laplacian_var",  "Sharpness (Laplacian var)", None),
    ("bright_p5",      "Brightness p5",            (0.0, 1.0)),
    ("bright_p50",     "Brightness p50",           (0.0, 1.0)),
    ("bright_p95",     "Brightness p95",           (0.0, 1.0)),
    ("highpass_ratio", "High-pass energy ratio",   (0.0, 0.5)),
    ("sat_mean",       "Saturation mean",          (0.0, 1.0)),
    ("sat_std",        "Saturation std",           (0.0, 0.5)),
    ("noise_residual", "Noise residual",           (0.0, 0.1)),
    ("edge_density",   "Edge density",             (0.0, 1.0)),
]


def print_image_stats_summary(all_img_sources: list[tuple[str, list[dict]]]):
    for source_name, records in all_img_sources:
        if not records:
            continue
        print(f"\n{'='*100}")
        print(f"  {source_name}  —  {len(records)} images")
        print(f"{'='*100}")
        for key, label, _ in IMAGE_STAT_PANELS:
            vals = np.array([r[key] for r in records])
            print(_prow(vals, label))


def plot_image_stats(all_img_sources: list[tuple[str, list[dict]]], output: Path,
                     bins: int = 40):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {name: SOURCE_PALETTE[i % len(SOURCE_PALETTE)]
              for i, (name, _) in enumerate(all_img_sources)}

    ncols, nrows = 4, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    fig.patch.set_facecolor("#111111")

    for idx, (key, label, xlim) in enumerate(IMAGE_STAT_PANELS):
        ax = axes[idx // ncols, idx % ncols]
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")

        for src_name, records in all_img_sources:
            vals = np.array([r[key] for r in records])
            if len(vals) == 0:
                continue
            if xlim is not None:
                vals = vals[vals <= xlim[1]]
            plot_range = (float(vals.min()), float(vals.max())) if xlim is None else xlim
            ax.hist(vals, bins=bins, range=plot_range, density=True,
                    color=colors[src_name], alpha=0.5, label=src_name)
            ax.axvline(float(np.median(vals)), color=colors[src_name],
                       linewidth=1.5, linestyle="--", alpha=0.9)

        ax.set_title(label, color="white", fontsize=9, fontweight="bold")
        ax.set_ylabel("Density", color="white", fontsize=7)
        ax.legend(facecolor="#222", labelcolor="white", fontsize=7)

    plt.tight_layout(pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, facecolor="#111111")
    plt.close(fig)
    print(f"Saved → {output}")


# ---------------------------------------------------------------------------
# MMD
# ---------------------------------------------------------------------------

FEATURE_KEYS = ["rel_w", "rel_h", "rel_area", "aspect_ratio", "cx_rel", "cy_rel"]


def _to_features(records: list[dict], cls_filter: str | None = None) -> np.ndarray:
    subset = records if cls_filter is None else [r for r in records if r["class"] == cls_filter]
    if not subset:
        return np.empty((0, len(FEATURE_KEYS)))
    return np.array([[r[k] for k in FEATURE_KEYS] for r in subset], dtype=np.float64)


def mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma: float | None = None,
            max_n: int = 2000) -> float:
    """
    Unbiased MMD² with RBF kernel. Returns MMD (not squared) for readability.
    Subsamples each set to max_n rows to keep memory bounded.
    """
    if len(X) == 0 or len(Y) == 0:
        return float("nan")

    rng = np.random.default_rng(0)
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]

    # Median heuristic for bandwidth if not given
    if sigma is None:
        combined = np.vstack([X, Y])
        sample = combined[rng.choice(len(combined), min(len(combined), 2000), replace=False)]
        dists = np.sum((sample[:, None] - sample[None, :]) ** 2, axis=-1)
        nonzero = dists[dists > 0]
        sigma = float(np.sqrt(np.median(nonzero) / 2)) if len(nonzero) else 1.0
    if sigma == 0:
        sigma = 1.0

    def rbf(A, B):
        d = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
        return np.exp(-d / (2 * sigma ** 2))

    n, m = len(X), len(Y)
    Kxx = rbf(X, X)
    np.fill_diagonal(Kxx, 0)
    Kyy = rbf(Y, Y)
    np.fill_diagonal(Kyy, 0)
    Kxy = rbf(X, Y)

    mmd2 = Kxx.sum() / (n * (n - 1)) - 2 * Kxy.mean() + Kyy.sum() / (m * (m - 1))
    return float(np.sqrt(max(mmd2, 0)))


def compute_mmd_per_experiment(
    base_dir: Path,
    loco_records: list[dict],
    cls_filter: str = "pallet_truck",
) -> list[tuple[str, int, float]]:
    """
    Compute MMD between each experiment's boxes and LOCO for a given class.
    Returns list of (exp_name, n_boxes, mmd) sorted by mmd descending.
    """
    from PIL import Image as PILImage

    ref_feats = _to_features(loco_records, cls_filter)
    # Use a shared sigma fitted on all base data + LOCO for fair comparison
    all_base = collect_base_stats(base_dir, source_label="_tmp")
    all_base_feats = _to_features(all_base, cls_filter)
    combined = np.vstack([all_base_feats, ref_feats]) if len(all_base_feats) else ref_feats
    rng = np.random.default_rng(0)
    sample = combined[rng.choice(len(combined), min(len(combined), 2000), replace=False)]
    dists = np.sum((sample[:, None] - sample[None, :]) ** 2, axis=-1)
    sigma = float(np.sqrt(np.median(dists[dists > 0]) / 2)) if dists[dists > 0].size else 1.0

    results = []
    for exp_dir in sorted(base_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        det_dir = exp_dir / "Camera" / "object_detection"
        if not det_dir.is_dir():
            continue
        exp_records = collect_base_stats(base_dir, source_label="_tmp", exp_filter=[exp_dir.name])
        feats = _to_features(exp_records, cls_filter)
        n = len(feats)
        mmd = mmd_rbf(feats, ref_feats, sigma=sigma) if n >= 5 else float("nan")
        results.append((exp_dir.name, n, mmd))

    results.sort(key=lambda t: (float("inf") if np.isnan(t[2]) else t[2]), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _prow(values: np.ndarray, label: str) -> str:
    if len(values) == 0:
        return f"  {label:30s}  N=0"
    p = np.percentile(values, [5, 25, 50, 75, 95])
    return (f"  {label:30s}  N={len(values):6d}  "
            f"mean={np.mean(values):.4f}  std={np.std(values):.4f}  "
            f"p5={p[0]:.4f}  p25={p[1]:.4f}  p50={p[2]:.4f}  p75={p[3]:.4f}  p95={p[4]:.4f}")


def print_summary(all_sources: list[tuple[str, list[dict]]], ref_name: str | None = None):
    """
    ref_name: source label to use as MMD reference. If given, every other source
              prints MMD distance against the reference (overall + per class).
    """
    metrics = ["rel_w", "rel_h", "rel_area", "aspect_ratio", "cx_rel", "cy_rel"]
    metric_labels = {
        "rel_w":         "rel width  (w/W)",
        "rel_h":         "rel height (h/H)",
        "rel_area":      "rel area (wh/WH)",
        "aspect_ratio":  "aspect ratio (w/h)",
        "cx_rel":        "center x (cx/W)",
        "cy_rel":        "center y (cy/H)",
    }

    # Pre-compute shared MMD bandwidth across all data for fair cross-source comparison
    ref_records = next((r for n, r in all_sources if n == ref_name), None) if ref_name else None
    shared_sigma: dict[str, float] = {}   # keyed by cls or "all"
    if ref_records:
        all_recs = [r for _, recs in all_sources for r in recs]
        rng = np.random.default_rng(0)
        for cls in [None] + CLASS_ORDER:
            feats = _to_features(all_recs, cls)
            if len(feats) > 1:
                # Subsample to cap pairwise matrix at ~2000 rows
                idx = rng.choice(len(feats), min(len(feats), 2000), replace=False)
                sample = feats[idx]
                dists = np.sum((sample[:, None] - sample[None, :]) ** 2, axis=-1)
                nonzero = dists[dists > 0]
                shared_sigma[cls or "all"] = float(np.sqrt(np.median(nonzero) / 2)) if len(nonzero) else 1.0

    for source_name, records in all_sources:
        if not records:
            continue
        print(f"\n{'='*100}")
        print(f"  {source_name}  —  {len(records)} boxes total")
        print(f"{'='*100}")

        cls_counts = Counter(r["class"] for r in records)
        total = sum(cls_counts.values())

        # Overall MMD vs reference
        if ref_records and source_name != ref_name:
            feats_all  = _to_features(records)
            ref_all    = _to_features(ref_records)
            mmd_all    = mmd_rbf(feats_all, ref_all, sigma=shared_sigma.get("all"))
            print(f"\n  MMD vs {ref_name}:  overall={mmd_all:.5f}")

        print("\n  Class distribution:")
        for cls in CLASS_ORDER:
            n = cls_counts.get(cls, 0)
            print(f"    {cls:20s}  {n:7d}  ({100*n/max(total,1):.1f}%)")

        for cls in CLASS_ORDER:
            cls_recs = [r for r in records if r["class"] == cls]
            if not cls_recs:
                continue

            # Per-class MMD vs reference
            mmd_str = ""
            if ref_records and source_name != ref_name:
                ref_cls = [r for r in ref_records if r["class"] == cls]
                if len(ref_cls) >= 5:
                    mmd_cls = mmd_rbf(_to_features(cls_recs), _to_features(ref_cls),
                                      sigma=shared_sigma.get(cls))
                    mmd_str = f"  MMD={mmd_cls:.5f}"

            print(f"\n  ── {cls} ({len(cls_recs)} boxes){mmd_str} ──")
            for metric in metrics:
                vals = np.array([r[metric] for r in cls_recs])
                print(_prow(vals, metric_labels[metric]))

        occ = np.array([r["occlusionRatio"] for r in records
                        if r.get("occlusionRatio") is not None and r["occlusionRatio"] >= 0])
        if len(occ):
            print(f"\n  ── occlusionRatio ──")
            print(_prow(occ, "occlusionRatio"))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_comparison(all_sources: list[tuple[str, list[dict]]], output: Path, bins: int = 40):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("rel_area",     "Relative area (wh/WH)",  (0, 0.5)),
        ("rel_w",        "Relative width (w/W)",    (0, 1.0)),
        ("rel_h",        "Relative height (h/H)",   (0, 1.0)),
        ("aspect_ratio", "Aspect ratio (w/h)",       (0, 10)),
        ("cx_rel",       "Center X (cx/W)",          (0, 1.0)),
        ("cy_rel",       "Center Y (cy/H)",          (0, 1.0)),
    ]

    colors = {name: SOURCE_PALETTE[i % len(SOURCE_PALETTE)] for i, (name, _) in enumerate(all_sources)}

    ncols = len(CLASS_ORDER) + 1
    nrows = len(metrics) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    fig.patch.set_facecolor("#111111")

    # ── Row 0: class distribution ──────────────────────────────────────────
    ax0 = axes[0, 0]
    ax0.set_facecolor("#1a1a1a")
    x = np.arange(len(CLASS_ORDER))
    bar_w = 0.8 / max(len(all_sources), 1)
    for i, (src_name, records) in enumerate(all_sources):
        total = len(records)
        fracs = [sum(1 for r in records if r["class"] == c) / max(total, 1) for c in CLASS_ORDER]
        offset = (i - (len(all_sources) - 1) / 2) * bar_w
        ax0.bar(x + offset, fracs, bar_w, label=src_name, color=colors[src_name], alpha=0.85)
    ax0.set_xticks(x)
    ax0.set_xticklabels([c.replace("_", "\n") for c in CLASS_ORDER], color="white", fontsize=8)
    ax0.set_ylabel("Fraction of boxes", color="white")
    ax0.tick_params(colors="white")
    ax0.spines[:].set_color("#444")
    ax0.legend(facecolor="#222", labelcolor="white", fontsize=8)
    ax0.set_title("Class distribution", color="white", fontsize=11, fontweight="bold")
    for col in range(1, ncols):
        axes[0, col].set_visible(False)

    col_titles = ["All classes"] + CLASS_ORDER

    # ── Rows 1..n: metric histograms ──────────────────────────────────────
    for row, (metric, label, xlim) in enumerate(metrics, start=1):
        for col, cls_filter in enumerate([None] + CLASS_ORDER):
            ax = axes[row, col]
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#444")

            if row == 1:
                ax.set_title(col_titles[col], color="white", fontsize=11, fontweight="bold")

            for src_name, records in all_sources:
                subset = records if cls_filter is None else [r for r in records if r["class"] == cls_filter]
                vals = np.array([r[metric] for r in subset])
                if len(vals) == 0:
                    continue
                clipped = vals[vals <= xlim[1]]
                ax.hist(clipped, bins=bins, range=xlim, density=True,
                        color=colors[src_name], alpha=0.5, label=src_name)
                ax.axvline(np.median(clipped), color=colors[src_name],
                           linewidth=1.5, linestyle="--", alpha=0.9)

            ax.set_xlabel(label, color="white", fontsize=8)
            ax.set_ylabel("Density", color="white", fontsize=8)
            ax.legend(facecolor="#222", labelcolor="white", fontsize=7)

    plt.tight_layout(pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, facecolor="#111111")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Statistical comparison: synth best trials / base run / LOCO real data"
    )
    parser.add_argument("--synth-root", default=None,
                        help="selected_trial_downloads/optuna-ec2/ (best_* Isaac Sim npy+json)")
    parser.add_argument("--base-dir", default=None,
                        help="palletjack_run_0/ or similar (KITTI-format object_detection txts)")
    parser.add_argument("--base-exps", nargs="+", default=None, metavar="EXP",
                        help="Filter base experiments by substring, e.g. --base-exps exp04 exp05")
    parser.add_argument("--mmd-rank", action="store_true",
                        help="Rank all base experiments by MMD distance to LOCO real data")
    parser.add_argument("--loco-ann", default=None,
                        help="LOCO COCO annotation JSON")
    parser.add_argument("--img-stats", action="store_true",
                        help="Compute and plot per-image pixel statistics")
    parser.add_argument("--output", default="output/stats_all_sources.png")
    parser.add_argument("--img-output", default="output/image_stats.png")
    parser.add_argument("--bins", type=int, default=40)
    args = parser.parse_args()

    all_sources = []

    if args.synth_root:
        print("Collecting synth (best trials) statistics ...")
        recs = collect_synth_stats(Path(args.synth_root), source_label="synth-best")
        print(f"  {len(recs)} boxes")
        all_sources.append(("synth-best", recs))

    if args.base_dir:
        label = f"base({','.join(args.base_exps)})" if args.base_exps else "base"
        print(f"Collecting base run statistics {('(filter: ' + str(args.base_exps) + ')') if args.base_exps else ''} ...")
        recs = collect_base_stats(Path(args.base_dir), source_label=label, exp_filter=args.base_exps)
        print(f"  {len(recs)} boxes  from: {sorted({r['exp'] for r in recs})}")
        all_sources.append((label, recs))

    if args.loco_ann:
        print("Collecting LOCO real statistics ...")
        recs = collect_loco_stats(Path(args.loco_ann), source_label="real-LOCO")
        print(f"  {len(recs)} boxes")
        all_sources.append(("real-LOCO", recs))

    if args.mmd_rank:
        if not args.base_dir or not args.loco_ann:
            parser.error("--mmd-rank requires both --base-dir and --loco-ann")
        loco_recs = collect_loco_stats(Path(args.loco_ann), source_label="real-LOCO")
        print(f"\nMMD ranking (pallet_truck vs LOCO, {len([r for r in loco_recs if r['class']=='pallet_truck'])} LOCO boxes):")
        print(f"  {'experiment':<45}  {'N':>5}  {'MMD':>8}")
        print(f"  {'-'*45}  {'-----':>5}  {'--------':>8}")
        for exp_name, n, mmd in compute_mmd_per_experiment(Path(args.base_dir), loco_recs):
            mmd_str = f"{mmd:.5f}" if not np.isnan(mmd) else "  (n<5)"
            print(f"  {exp_name:<45}  {n:>5}  {mmd_str:>8}")
        return

    if not all_sources:
        print("No data sources provided. Use --synth-root, --base-dir, or --loco-ann.")
        return

    ref_name = "real-LOCO" if any(n == "real-LOCO" for n, _ in all_sources) else None
    print_summary(all_sources, ref_name=ref_name)

    print("\nGenerating plots ...")
    plot_comparison(all_sources, Path(args.output), bins=args.bins)
    print(f"Saved → {args.output}")

    if args.img_stats:
        all_img_sources = []
        if args.loco_ann:
            loco_img_root = Path(args.loco_ann).parents[2] / "loco_dataset"
            print(f"\nComputing LOCO image statistics (images: {loco_img_root}) ...")
            img_stats = collect_loco_image_stats(
                Path(args.loco_ann), loco_img_root, source_label="real-LOCO")
            print(f"  {len(img_stats)} images")
            all_img_sources.append(("real-LOCO", img_stats))
        if args.synth_root:
            print("Computing synth image statistics ...")
            img_stats = collect_synth_image_stats(
                Path(args.synth_root), source_label="synth-best")
            print(f"  {len(img_stats)} images")
            all_img_sources.append(("synth-best", img_stats))
        if args.base_dir:
            label = f"base({','.join(args.base_exps)})" if args.base_exps else "base"
            print(f"Computing base image statistics ...")
            img_stats = collect_base_image_stats(
                Path(args.base_dir), source_label=label, exp_filter=args.base_exps)
            print(f"  {len(img_stats)} images")
            all_img_sources.append((label, img_stats))
        if all_img_sources:
            print_image_stats_summary(all_img_sources)
            print("\nGenerating image stats plots ...")
            plot_image_stats(all_img_sources, Path(args.img_output), bins=args.bins)


if __name__ == "__main__":
    main()
