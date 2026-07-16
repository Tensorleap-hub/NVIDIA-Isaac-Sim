"""Per-run MMD table: every Cosmos clip vs its Isaac parent, both vs the real LOCO pool.

Each Cosmos-Transfer output clip has exactly one Isaac trajectory-SDG "parent" (the render
it was conditioned on, resolved via the sidecar's ``video_path`` basename). This script
builds a per-run table:

    run (cosmos clip) | isaac parent | n | MMD(isaac, real) | MMD(cosmos, real) | delta

- MMD(cosmos, real): how far the stylized clip sits from real footage.
- MMD(isaac, real):  how far its raw Isaac parent sits from real footage.
- delta = MMD(isaac) - MMD(cosmos): positive => Cosmos moved this run *toward* real.

Both distributions for a run use the SAME frame indices (Cosmos output is frame-for-frame
aligned with its parent's Camera/rgb), so the pair is directly comparable. A single RBF
gamma (median heuristic on the real pool) is reused for every MMD, matching
compare_cosmos_mmd.py and the Optuna loop. Multiple Cosmos clips can share one Isaac parent
(e.g. the clean/gritty prompt variants of the same seed) -- the parent's MMD is computed
once and reused.

Usage (loop venv has cv2 + cached DINOv2 weights):
    .sim_loop_venv/bin/python cosmos_v4_optuna_track/cosmos_mmd_per_run.py \
        --cosmos-dirs palletjack_sdg/palletjack_data/trajectory/cosmos_transfer/20260712_cosmos_transfer_all_seeds_2prompts \
                      palletjack_sdg/palletjack_data/trajectory/cosmos_transfer/20260713_cosmos_transfer_new_themes_2seeds \
                      palletjack_sdg/palletjack_data/trajectory/cosmos_transfer/20260713_cosmos_transfer_optuna_23runs_1seed \
        --source-roots palletjack_sdg/palletjack_data/trajectory/cosmos_v4_20260712_194849 \
                       palletjack_sdg/palletjack_data/trajectory/cosmos_optuna_20260713_025849 \
        --frame-stride 4 \
        --csv cosmos_v4_optuna_track/cosmos_mmd_per_run.csv \
        --report cosmos_v4_optuna_track/cosmos_mmd_per_run.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cosmos_v4_optuna_track"))

import cv2  # noqa: E402
import compare_cosmos_mmd as ccm  # noqa: E402
from calibration_optuna.metrics import DistributionMetrics  # noqa: E402
from simulation_calibration_loop.data import DINOv2Embedder, select_real_image_paths  # noqa: E402


def resolve_isaac_parent(sidecar: dict, source_roots: list[Path]) -> Path | None:
    """The sidecar video_path is <exp>/video/clip_XXXX/rgb.mp4; re-home <exp> by basename."""
    vp = sidecar.get("video_path")
    if not vp:
        return None
    recorded = Path(vp).parents[2]
    if recorded.is_dir():
        return recorded
    for root in source_roots:
        cand = root / recorded.name
        if cand.is_dir():
            return cand
    return None


def decode_clip_frames(mp4: Path, cache_dir: Path, stride: int) -> tuple[list[Path], list[int]]:
    """Decode every ``stride``-th frame of one mp4 to PNGs; return (paths, frame_indices)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        return [], []
    paths, idxs, i = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            out = cache_dir / f"{mp4.stem}_{i:04d}.png"
            if not out.exists():
                cv2.imwrite(str(out), frame)
            paths.append(out)
            idxs.append(i)
        i += 1
    cap.release()
    return paths, idxs


def isaac_frames_for_indices(parent: Path, idxs: list[int]) -> list[Path]:
    """Original Camera/rgb frames at the given indices (aligned with the Cosmos output)."""
    rgb_dir = parent / "Camera" / "rgb"
    return [rgb_dir / f"rgb_{i:04d}.png" for i in idxs if (rgb_dir / f"rgb_{i:04d}.png").exists()]


def find_cosmos_clips(cosmos_dirs: list[Path]) -> list[Path]:
    clips = []
    for d in cosmos_dirs:
        if not d.is_dir():
            print(f"  SKIP (not a dir): {d}")
            continue
        for mp4 in sorted(d.glob("*.mp4")):
            if "_control_" in mp4.name:
                continue
            if mp4.with_suffix(".json").exists():
                clips.append(mp4)
    return clips


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-run MMD: each Cosmos clip vs its Isaac parent vs real")
    ap.add_argument("--cosmos-dirs", nargs="+", required=True, help="Cosmos-Transfer output dirs")
    ap.add_argument("--source-roots", nargs="+", required=True,
                    help="Local dirs holding Isaac parent runs (searched by basename)")
    ap.add_argument("--real-root", default=ccm.DEFAULT_REAL_ROOT)
    ap.add_argument("--real-annotations", default=ccm.DEFAULT_REAL_ANNOTATIONS)
    ap.add_argument("--frame-stride", type=int, default=4,
                    help="Keep every Nth frame per clip (default 4 => 32 frames/128-frame clip)")
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "od_scripts" / "data" / "cosmos_mmd_per_run"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    cosmos_dirs = [Path(d) for d in args.cosmos_dirs]
    source_roots = [Path(d) for d in args.source_roots]
    cache_root = Path(args.cache_dir)
    frames_cache = cache_root / "frames"
    embed_cache = cache_root / "embeddings"
    embed_cache.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if ccm._cuda_available() else "cpu")
    print(f"Embedder: DINOv2 {ccm.DINOV2_MODEL} on {device}")

    embedder = DINOv2Embedder(ccm.DINOV2_REPO, ccm.DINOV2_MODEL, device, ccm.IMAGE_SIZE, ccm.RESIZE_SIZE)

    # Real reference + shared gamma
    real_paths = select_real_image_paths(args.real_root, args.real_annotations)
    if not real_paths:
        raise SystemExit("No real images resolved")
    real_emb = ccm.embed(embedder, real_paths, embed_cache / "real.npy")
    gamma = DistributionMetrics._compute_gamma_median_heuristic(real_emb, real_emb)
    print(f"Real reference: {len(real_emb)} images | shared RBF gamma {gamma:.6g}\n")

    clips = find_cosmos_clips(cosmos_dirs)
    print(f"Found {len(clips)} cosmos clips\n")

    parent_mmd_cache: dict[str, float] = {}  # parent name -> MMD(isaac, real)
    rows = []
    for mp4 in clips:
        sidecar = json.loads(mp4.with_suffix(".json").read_text())
        parent = resolve_isaac_parent(sidecar, source_roots)
        if parent is None:
            print(f"  SKIP (no isaac parent): {mp4.name}")
            continue

        cos_paths, idxs = decode_clip_frames(mp4, frames_cache / mp4.parent.name, args.frame_stride)
        isaac_paths = isaac_frames_for_indices(parent, idxs)
        if not cos_paths or not isaac_paths:
            print(f"  SKIP (no frames): {mp4.name}")
            continue

        cos_emb = ccm.embed(embedder, cos_paths, embed_cache / f"cos_{mp4.stem}.npy")
        mmd_cos = DistributionMetrics.mmd(cos_emb, real_emb, gamma=gamma)

        if parent.name not in parent_mmd_cache:
            isaac_emb = ccm.embed(embedder, isaac_paths, embed_cache / f"isaac_{parent.name}.npy")
            parent_mmd_cache[parent.name] = DistributionMetrics.mmd(isaac_emb, real_emb, gamma=gamma)
        mmd_isaac = parent_mmd_cache[parent.name]

        rows.append({
            "run": mp4.stem,
            "isaac_parent": parent.name,
            "set": mp4.parent.name,
            "n_frames": len(cos_paths),
            "mmd_isaac_vs_real": round(mmd_isaac, 6),
            "mmd_cosmos_vs_real": round(mmd_cos, 6),
            "delta_isaac_minus_cosmos": round(mmd_isaac - mmd_cos, 6),
        })
        print(f"  {mp4.stem:52s} isaac {mmd_isaac:.4f}  cosmos {mmd_cos:.4f}  d {mmd_isaac - mmd_cos:+.4f}")

    rows.sort(key=lambda r: r["mmd_cosmos_vs_real"])

    # ---- Print table ----
    print("\n" + "=" * 118)
    print(f"{'run':52s} {'set':10s} {'n':>3s} {'isaac→real':>11s} {'cosmos→real':>12s} {'Δ(i-c)':>8s}")
    print("-" * 118)
    for r in rows:
        setshort = r["set"].replace("_cosmos_transfer", "").replace("20260712_", "").replace("20260713_", "")[:10]
        print(f"{r['run']:52s} {setshort:10s} {r['n_frames']:>3d} "
              f"{r['mmd_isaac_vs_real']:>11.4f} {r['mmd_cosmos_vs_real']:>12.4f} "
              f"{r['delta_isaac_minus_cosmos']:>+8.4f}")
    print("=" * 118)

    if rows:
        ci = np.array([r["mmd_cosmos_vs_real"] for r in rows])
        ii = np.array([r["mmd_isaac_vs_real"] for r in rows])
        moved = int((ci < ii).sum())
        print(f"\n{len(rows)} runs | mean MMD→real: isaac {ii.mean():.4f}, cosmos {ci.mean():.4f} "
              f"| Cosmos moved {moved}/{len(rows)} runs toward real")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"CSV written to {args.csv}")
    if args.report:
        Path(args.report).write_text(json.dumps({
            "embedder": {"backend": "dinov2", "model": ccm.DINOV2_MODEL},
            "real": {"root": args.real_root, "annotations": args.real_annotations, "n": len(real_emb)},
            "frame_stride": args.frame_stride,
            "rbf_gamma": float(gamma),
            "runs": rows,
        }, indent=2))
        print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
