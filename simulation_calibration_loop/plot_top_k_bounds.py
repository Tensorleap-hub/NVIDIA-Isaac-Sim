"""Identify parameter bounds by comparing top-k% runs vs the rest.

For each parameter found across state files:
  - Numeric:     Mann-Whitney U test + suggested bound from top-k IQR/percentiles
  - Categorical: Chi-square test + which categories dominate the top-k

Produces one subplot per significant parameter showing the distribution split.

Usage:
    python plot_top_k_bounds.py <state_dir> [--top-pct 20] [--pvalue 0.05] [--out <png>]
                                            [--save-bounds <bounds.yaml>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import yaml
from scipy import stats


# ── data loading ───────────────────────────────────────────────────────────────

def load_all_artifacts(state_dir: Path) -> list[dict]:
    """Return flat list of {param: value, ..., _mmd, _theme} for every artifact."""
    records = []
    for path in sorted(state_dir.glob("*.json")):
        data = json.loads(path.read_text())
        theme = path.stem.replace("_state", "")
        for iteration in data.get("iterations", []):
            for art in iteration.get("artifacts", []):
                mmd = art.get("objective_value")
                if mmd is None:
                    continue
                rec = dict(art.get("flattened_params", {}))
                rec["_mmd"] = float(mmd)
                rec["_theme"] = theme
                records.append(rec)
    return records


def collect_param_values(records: list[dict], param: str) -> tuple[list, list[float]]:
    """Extract (raw_value, mmd) pairs for a param, skipping null/missing."""
    vals, mmds = [], []
    for rec in records:
        raw = rec.get(param)
        if raw in (None, "null", "None", ""):
            continue
        vals.append(raw)
        mmds.append(rec["_mmd"])
    return vals, mmds


def coerce_numeric(vals: list) -> tuple[bool, np.ndarray | list]:
    try:
        return True, np.array([float(v) for v in vals])
    except (ValueError, TypeError):
        return False, vals


# ── statistical tests ──────────────────────────────────────────────────────────

def test_numeric(top_vals: np.ndarray, rest_vals: np.ndarray) -> tuple[float, float]:
    """Mann-Whitney U: returns (statistic, p_value). Lower MMD = better, so we
    test whether top-k param values are drawn from a different distribution."""
    if len(top_vals) < 2 or len(rest_vals) < 2:
        return float("nan"), 1.0
    stat, p = stats.mannwhitneyu(top_vals, rest_vals, alternative="two-sided")
    return float(stat), float(p)


def test_categorical(top_labels: list, rest_labels: list) -> tuple[float, float]:
    """Chi-square test on category counts: top-k vs rest."""
    cats = sorted(set(top_labels) | set(rest_labels))
    top_counts  = np.array([top_labels.count(c)  for c in cats], dtype=float)
    rest_counts = np.array([rest_labels.count(c) for c in cats], dtype=float)
    # Need expected >= 5 in each cell; collapse rare categories
    if any(top_counts + rest_counts < 5):
        pass  # still run; chi2 will be approximate
    observed = np.vstack([top_counts, rest_counts])
    if observed.shape[1] < 2:
        return float("nan"), 1.0
    chi2, p, _, _ = stats.chi2_contingency(observed)
    return float(chi2), float(p)


# ── bound suggestion ───────────────────────────────────────────────────────────

def suggest_numeric_bound(top_vals: np.ndarray, all_vals: np.ndarray, pct: tuple[float,float] = (10, 90)) -> dict:
    lo, hi = np.percentile(top_vals, pct[0]), np.percentile(top_vals, pct[1])
    all_lo, all_hi = all_vals.min(), all_vals.max()
    tightened = (hi - lo) / (all_hi - all_lo + 1e-12) < 0.85
    return {
        "current_range": (float(all_lo), float(all_hi)),
        "suggested_range": (float(lo), float(hi)),
        "tightened": tightened,
        "percentile_band": pct,
        "top_mean": float(top_vals.mean()),
        "top_std":  float(top_vals.std()),
        "all_mean": float(all_vals.mean()),
        "all_std":  float(all_vals.std()),
    }


def suggest_categorical_bound(top_labels: list, all_labels: list) -> dict:
    cats = sorted(set(all_labels))
    total_top = len(top_labels)
    total_all = len(all_labels)
    rows = []
    for c in cats:
        n_top = top_labels.count(c)
        n_all = all_labels.count(c)
        top_rate = n_top / total_top if total_top else 0
        base_rate = n_all / total_all if total_all else 0
        lift = top_rate / base_rate if base_rate > 0 else float("inf")
        rows.append({"category": c, "top_rate": top_rate, "base_rate": base_rate,
                     "lift": lift, "n_top": n_top, "n_all": n_all})
    rows.sort(key=lambda r: r["lift"], reverse=True)
    keep = [r["category"] for r in rows if r["lift"] >= 1.2]
    return {"categories": rows, "suggested_keep": keep}


# ── plotting ───────────────────────────────────────────────────────────────────

def _plot_numeric(ax: plt.Axes, top_vals: np.ndarray, rest_vals: np.ndarray,
                  bound: dict, param: str, p_val: float) -> None:
    bins = np.linspace(
        min(top_vals.min(), rest_vals.min()),
        max(top_vals.max(), rest_vals.max()),
        25,
    )
    ax.hist(rest_vals, bins=bins, alpha=0.45, color="steelblue", label="rest", density=True)
    ax.hist(top_vals,  bins=bins, alpha=0.65, color="tomato",    label="top-k", density=True)
    lo, hi = bound["suggested_range"]
    ax.axvline(lo, color="tomato", linestyle="--", linewidth=1.4)
    ax.axvline(hi, color="tomato", linestyle="--", linewidth=1.4)
    ax.axvspan(lo, hi, alpha=0.08, color="tomato", label=f"suggested [{lo:.3g}, {hi:.3g}]")
    ax.set_title(f"{param}\np={p_val:.3g}", fontsize=8, fontweight="bold")
    ax.set_xlabel(param, fontsize=7)
    ax.set_ylabel("density", fontsize=7)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.25, axis="y")


def _plot_categorical(ax: plt.Axes, top_labels: list, rest_labels: list,
                      bound: dict, param: str, p_val: float) -> None:
    cats = [r["category"] for r in bound["categories"]]
    top_rates  = [r["top_rate"]  for r in bound["categories"]]
    base_rates = [r["base_rate"] for r in bound["categories"]]
    x = np.arange(len(cats))
    w = 0.38
    bars_rest = ax.bar(x - w/2, base_rates, w, alpha=0.55, color="steelblue", label="base rate")
    bars_top  = ax.bar(x + w/2, top_rates,  w, alpha=0.80, color="tomato",    label="top-k rate")
    keep = set(bound["suggested_keep"])
    for bar, cat in zip(bars_top, cats):
        if cat in keep:
            bar.set_edgecolor("gold")
            bar.set_linewidth(2)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=6)
    ax.set_title(f"{param}\np={p_val:.3g}", fontsize=8, fontweight="bold")
    ax.set_ylabel("proportion", fontsize=7)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.25, axis="y")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Top-k parameter bounding analysis")
    parser.add_argument("state_dir", help="Directory containing *_state.json files")
    parser.add_argument("--top-pct", type=float, default=20.0,
                        help="Percentage of runs to treat as 'top-k' (default: 20)")
    parser.add_argument("--pvalue", type=float, default=0.05,
                        help="Significance threshold (default: 0.05)")
    parser.add_argument("--bound-pct", type=float, nargs=2, default=[10, 90],
                        metavar=("LO", "HI"),
                        help="Percentile band for numeric bound suggestion (default: 10 90)")
    parser.add_argument("--out", default=None, metavar="PNG")
    parser.add_argument("--save-bounds", default=None, metavar="YAML",
                        help="Save significant bounds to a YAML file for use with apply_bounds.py")
    args = parser.parse_args()

    state_dir = Path(args.state_dir).expanduser().resolve()
    if not state_dir.is_dir():
        sys.exit(f"Not a directory: {state_dir}")

    records = load_all_artifacts(state_dir)
    if not records:
        sys.exit("No artifacts found.")

    mmds = np.array([r["_mmd"] for r in records])
    threshold = np.percentile(mmds, args.top_pct)
    top_mask = mmds <= threshold
    n_top = top_mask.sum()
    print(f"Total runs: {len(records)}  |  top {args.top_pct:.0f}%: {n_top}  (MMD ≤ {threshold:.4f})")

    all_params = sorted({k for r in records for k in r if not k.startswith("_")})

    results = []
    for param in all_params:
        all_vals, all_mmds_param = collect_param_values(records, param)
        if len(all_vals) < 10:
            continue
        is_numeric, coerced = coerce_numeric(all_vals)

        # align top mask to this param's subset
        param_indices = [i for i, r in enumerate(records)
                         if r.get(param) not in (None, "null", "None", "")]
        param_top_mask = top_mask[param_indices]
        if param_top_mask.sum() < 2 or (~param_top_mask).sum() < 2:
            continue

        if is_numeric:
            arr = coerced
            top_v  = arr[param_top_mask]
            rest_v = arr[~param_top_mask]
            stat, p = test_numeric(top_v, rest_v)
            bound = suggest_numeric_bound(top_v, arr, tuple(args.bound_pct))
            results.append({
                "param": param, "is_numeric": True, "p_value": p,
                "top_vals": top_v, "rest_vals": rest_v,
                "all_vals": arr, "bound": bound,
            })
        else:
            labels = list(coerced)
            top_labels  = [labels[i] for i, m in enumerate(param_top_mask) if m]
            rest_labels = [labels[i] for i, m in enumerate(param_top_mask) if not m]
            stat, p = test_categorical(top_labels, rest_labels)
            bound = suggest_categorical_bound(top_labels, labels)
            results.append({
                "param": param, "is_numeric": False, "p_value": p,
                "top_labels": top_labels, "rest_labels": rest_labels,
                "all_labels": labels, "bound": bound,
            })

    significant = [r for r in results if r["p_value"] <= args.pvalue]
    significant.sort(key=lambda r: r["p_value"])

    # ── console report ─────────────────────────────────────────────────────────
    print(f"\nSignificant params (p ≤ {args.pvalue}):  {len(significant)} / {len(results)}\n")
    print(f"{'param':<40} {'type':<10} {'p-value':>9}  suggestion")
    print("-" * 100)
    for r in significant:
        param = r["param"]
        p = r["p_value"]
        if r["is_numeric"]:
            b = r["bound"]
            tightened = "✓ tightened" if b["tightened"] else "  (minor)"
            lo, hi = b["suggested_range"]
            print(f"{param:<40} {'numeric':<10} {p:>9.4f}  [{lo:.3g}, {hi:.3g}]  {tightened}")
        else:
            keep = r["bound"]["suggested_keep"]
            print(f"{param:<40} {'categoric':<10} {p:>9.4f}  keep: {keep}")
    print()

    # ── plot ───────────────────────────────────────────────────────────────────
    if not significant:
        print("Nothing significant to plot.")
        return

    n = len(significant)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    fig.suptitle(
        f"Top-{args.top_pct:.0f}% vs rest  (MMD ≤ {threshold:.4f},  n_top={n_top})",
        fontsize=11, fontweight="bold",
    )

    for ax, r in zip(axes, significant):
        if r["is_numeric"]:
            _plot_numeric(ax, r["top_vals"], r["rest_vals"], r["bound"], r["param"], r["p_value"])
        else:
            _plot_categorical(ax, r["top_labels"], r["rest_labels"], r["bound"], r["param"], r["p_value"])

    for ax in axes[len(significant):]:
        ax.set_visible(False)

    plt.tight_layout()
    out_path = Path(args.out) if args.out else state_dir / f"top_k_bounds_{args.top_pct:.0f}pct.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if args.save_bounds:
        save_bounds_yaml(significant, Path(args.save_bounds), args.top_pct, args.pvalue, threshold)


def save_bounds_yaml(significant: list[dict], out_path: Path, top_pct: float, pvalue: float, threshold: float) -> None:
    params_block = {}
    for r in significant:
        param = r["param"]
        if r["is_numeric"]:
            lo, hi = r["bound"]["suggested_range"]
            params_block[param] = {
                "type": "numeric",
                "min": round(float(lo), 6),
                "max": round(float(hi), 6),
                "p_value": round(float(r["p_value"]), 6),
            }
        else:
            keep = r["bound"]["suggested_keep"]
            # Resolve list-valued categoricals back to native Python types
            resolved = []
            for v in keep:
                try:
                    resolved.append(yaml.safe_load(v))
                except Exception:
                    resolved.append(v)
            params_block[param] = {
                "type": "categorical",
                "keep": resolved,
                "p_value": round(float(r["p_value"]), 6),
            }

    doc = {
        "meta": {
            "top_pct": top_pct,
            "pvalue_threshold": pvalue,
            "mmd_threshold": round(float(threshold), 6),
            "n_significant": len(significant),
        },
        "params": params_block,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.dump(doc, sort_keys=False, allow_unicode=True))
    print(f"Bounds saved: {out_path}")


if __name__ == "__main__":
    main()
