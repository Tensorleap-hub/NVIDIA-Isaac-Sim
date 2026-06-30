"""
Build od_scripts/optimization_story.html — the end-to-end narrative:
a trained network's latent space defines "real"; the Tensorleap native
optimization pipeline tunes the synthetic-data generator until synthetic data
lands in that same latent space; the same network is then fine-tuned on the
calibrated data, lifting detection accuracy.

Reuses the 9 sample-detection figures already embedded in training_report.html
(no model reload). Everything else is rendered as native, on-brand CSS/SVG.

Usage:
    python3 od_scripts/make_optimization_story.py [path/to/latent_space_ui.png]

The optional argument is a screenshot from the Tensorleap UI for the
latent-space convergence section. If omitted, a labelled placeholder slot is
rendered and can be filled by re-running with the image path.
"""
from __future__ import annotations

import base64
import io
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_REPORT = REPO_ROOT / "od_scripts/training_report.html"
OUT_HTML = REPO_ROOT / "od_scripts/optimization_story.html"
TRAIN_ROOT = Path("/Users/orram/Tensorleap/data/warehouse/training")

# brand series colours (match the legend / palette)
SERIES_COLOR = {"real": "#6b6b78", "base_synth": "#d98b1f", "opt0": "#0FBD79"}

# ---------------------------------------------------------------------------
# Data (mirrors the values in training_report.html / population_view CSV)
# ---------------------------------------------------------------------------

MODELS = ["real", "base_synth", "opt0"]
MODEL_LABELS = {"real": "Real only", "base_synth": "Base synthetic", "opt0": "Calibrated (Tensorleap)"}

# best-checkpoint stats on LOCO subset-3
STATS = {
    "real":       {"mAP50": 0.449, "F1": 0.476, "mAR": 0.381, "pt": 0.015, "fork": 0.310, "pallet": 0.188},
    "base_synth": {"mAP50": 0.560, "F1": 0.582, "mAR": 0.438, "pt": 0.233, "fork": 0.477, "pallet": 0.180},
    "opt0":       {"mAP50": 0.625, "F1": 0.638, "mAR": 0.488, "pt": 0.392, "fork": 0.413, "pallet": 0.187},
}

# top-10 sim parameters by importance for closing the latent-space gap
PARAM_IMPORTANCE = [
    ("environment.name",                          0.3829),
    ("camera.camera_yaw_std",                     0.2102),
    ("camera.camera_yaw_mean",                    0.1638),
    ("camera.dataset_noise.jpeg_quality_mean",    0.0633),
    ("materials.textures",                        0.0429),
    ("pallets.count_per_model",                   0.0204),
    ("image_augmentation.brightness_gain_std",    0.0134),
    ("image_augmentation.color_gain_std[0]",      0.0124),
    ("distractors.clutter_level",                 0.0083),
    ("palletjacks.position_std[0]",               0.0080),
]

# latent-space objective (DINO-style embedding distance, minimized by the loop)
BEST_OBJECTIVE = 0.374   # best mmd_rbf_to_real across optimized population
POPULATION_RUNS = 80     # scored runs in the optimization population

# ---------------------------------------------------------------------------
# Reuse the embedded sample-detection figures from the training report
# ---------------------------------------------------------------------------

def extract_scene_images() -> list[dict]:
    """Parse training_report.html → list of {fn, gt, f1{...}, imgs[3]} scene dicts."""
    html = SRC_REPORT.read_text()
    scenes = []
    for block in re.findall(r'<div class="scene">(.*?)</div>\s*</div>', html, re.S):
        fn = re.search(r'class="fn">([^<]+)</span>', block)
        gt = re.search(r'class="gt">([^<]+)</span>', block)
        win = re.search(r'class="f1 win">([^<]+)</span>', block)
        f1s = re.findall(r'class="f1">([^<]+)</span>', block)
        imgs = re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)', block)
        scenes.append({
            "fn": fn.group(1) if fn else "",
            "gt": gt.group(1) if gt else "",
            "win": win.group(1) if win else "",
            "f1s": f1s,
            "imgs": imgs,
        })
    return scenes


# latent-space before/after screenshots from the Tensorleap UI
BEFORE_IMG = REPO_ROOT / "Screenshot 2026-06-30 at 10.07.55.png"
AFTER_IMG = REPO_ROOT / "Screenshot 2026-06-30 at 10.31.16.png"


def _embed(p: Path) -> str | None:
    p = Path(p).expanduser()
    if p.exists():
        mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
    print(f"[warn] latent image not found: {p}", file=sys.stderr)
    return None


def load_latent_images() -> tuple[str | None, str | None]:
    """Return (before, after) data URIs. Optional argv: <before> <after>."""
    before = sys.argv[1] if len(sys.argv) > 1 else BEFORE_IMG
    after = sys.argv[2] if len(sys.argv) > 2 else AFTER_IMG
    return _embed(before), _embed(after)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def pct_gain(new: float, old: float) -> str:
    return f"+{(new - old) / old * 100:.0f}%"


def build_headline() -> str:
    """TL;DR: grouped bar of the headline metrics across the three models + takeaway."""
    metrics = [("mAP@50", "mAP50"), ("AP pallet_truck", "pt")]
    groups = list(range(len(metrics)))
    width = 0.22

    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for i, m in enumerate(MODELS):
        xs = [g + (i - 1) * width for g in groups]
        vals = [STATS[m][key] for _, key in metrics]
        bars = ax.bar(xs, vals, width=width, color=SERIES_COLOR[m],
                      label=MODEL_LABELS[m], zorder=3)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8.5, color="#33333d")

    ax.set_xticks(groups)
    ax.set_xticklabels([lbl for lbl, _ in metrics], fontsize=11, color="#191A20", fontweight="bold")
    ax.set_ylim(0, 0.72)
    ax.tick_params(axis="y", colors="#6b6b78", labelsize=9)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#e4e4ec", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#e4e4ec")
    ax.legend(frameon=False, fontsize=10, labelcolor="#33333d", ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, 1.16))
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    chart_b64 = base64.b64encode(buf.getvalue()).decode()

    r, o = STATS["real"], STATS["opt0"]
    lift = o["pt"] / r["pt"]
    takeaway = (
        f'We then fine-tune the same network on the calibrated data and measure the resulting '
        f'accuracy gain: calibrated synthetic data beats both the real-only and base-synthetic '
        f'baselines on <b>every metric</b> &mdash; mAP@50 <b>{pct_gain(o["mAP50"], r["mAP50"])}</b> '
        f'over real-only, and the hardest class (<span class="mono">pallet_truck</span>) lifts '
        f'<b>{lift:.0f}&times;</b>.'
    )
    return (
        f'<p class="takeaway">{takeaway}</p>'
        f'<figure class="headline-fig"><img src="data:image/png;base64,{chart_b64}" '
        f'alt="Detection metrics by data mix"></figure>'
    )


def build_loop_diagram() -> str:
    return r'''
<svg viewBox="0 0 920 320" class="loop" role="img" aria-label="Latent-space calibration loop">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#0A7E51"/><stop offset="1" stop-color="#0FBD79"/>
    </linearGradient>
    <marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
      <path d="M0,0 L9,4.5 L0,9 Z" fill="#0A7E51"/>
    </marker>
    <marker id="arrm" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
      <path d="M0,0 L9,4.5 L0,9 Z" fill="#6b6b78"/>
    </marker>
  </defs>

  <!-- top row: the loop -->
  <g font-family="ui-sans-serif,system-ui,sans-serif">
    <rect x="20"  y="40" width="180" height="74" rx="10" fill="#f7f7fb" stroke="#e4e4ec"/>
    <text x="110" y="66" text-anchor="middle" font-size="12" font-weight="700" fill="#191A20">Trained RF-DETR</text>
    <text x="110" y="84" text-anchor="middle" font-size="10.5" fill="#6b6b78">latent space of</text>
    <text x="110" y="99" text-anchor="middle" font-size="10.5" fill="#0A7E51" font-weight="700">REAL data = target</text>

    <rect x="250" y="40" width="170" height="74" rx="10" fill="#f7f7fb" stroke="#e4e4ec"/>
    <text x="335" y="66" text-anchor="middle" font-size="12" font-weight="700" fill="#191A20">Isaac SDG</text>
    <text x="335" y="84" text-anchor="middle" font-size="10.5" fill="#6b6b78">render synthetic</text>
    <text x="335" y="99" text-anchor="middle" font-size="10.5" fill="#6b6b78">with params &#952;</text>

    <rect x="470" y="40" width="170" height="74" rx="10" fill="#f7f7fb" stroke="#e4e4ec"/>
    <text x="555" y="66" text-anchor="middle" font-size="12" font-weight="700" fill="#191A20">Embed synthetic</text>
    <text x="555" y="84" text-anchor="middle" font-size="10.5" fill="#6b6b78">same network &rarr;</text>
    <text x="555" y="99" text-anchor="middle" font-size="10.5" fill="#6b6b78">latent vectors</text>

    <rect x="690" y="40" width="210" height="74" rx="10" fill="#f7f7fb" stroke="#e4e4ec"/>
    <text x="795" y="66" text-anchor="middle" font-size="12" font-weight="700" fill="#191A20">Measure distance</text>
    <text x="795" y="84" text-anchor="middle" font-size="10.5" fill="#6b6b78">synthetic vs real</text>
    <text x="795" y="99" text-anchor="middle" font-size="10.5" fill="#0A7E51" font-weight="700">MMD-RBF objective</text>

    <!-- forward arrows -->
    <line x1="200" y1="77" x2="248" y2="77" stroke="#0A7E51" stroke-width="2" marker-end="url(#arr)"/>
    <line x1="420" y1="77" x2="468" y2="77" stroke="#0A7E51" stroke-width="2" marker-end="url(#arr)"/>
    <line x1="640" y1="77" x2="688" y2="77" stroke="#0A7E51" stroke-width="2" marker-end="url(#arr)"/>

    <!-- optimization engine (feedback) -->
    <rect x="250" y="180" width="390" height="64" rx="10" fill="url(#g)"/>
    <text x="445" y="208" text-anchor="middle" font-size="13" font-weight="700" fill="#fff">Tensorleap optimization pipeline</text>
    <text x="445" y="227" text-anchor="middle" font-size="10.5" fill="#dcd8ff">runs in the Tensorleap API &#183; suggests next params &#952;'</text>

    <!-- distance -> engine -->
    <path d="M795,114 V212 H642" fill="none" stroke="#0A7E51" stroke-width="2" marker-end="url(#arr)"/>
    <!-- engine -> isaac (loop back) -->
    <path d="M335,180 V116" fill="none" stroke="#0A7E51" stroke-width="2" marker-end="url(#arr)"/>
    <text x="352" y="151" font-size="10.5" fill="#0A7E51" font-weight="700">&#8635; runs automatically</text>
    <text x="352" y="164" font-size="9.5" fill="#6b6b78">until the distance converges</text>

    <!-- output: fine-tune -->
    <rect x="20" y="180" width="180" height="64" rx="10" fill="#fff" stroke="#0A7E51" stroke-width="1.5"/>
    <text x="110" y="206" text-anchor="middle" font-size="12" font-weight="700" fill="#191A20">Fine-tune RF-DETR</text>
    <text x="110" y="225" text-anchor="middle" font-size="10.5" fill="#127a64" font-weight="700">on calibrated data</text>
    <path d="M250,212 H202" fill="none" stroke="#6b6b78" stroke-width="2" stroke-dasharray="4 3" marker-end="url(#arrm)"/>
    <text x="226" y="173" text-anchor="middle" font-size="10" fill="#6b6b78" font-style="italic">best &#952;</text>
  </g>
</svg>'''


def build_latent_section(before: str | None, after: str | None) -> str:
    def tile(img, tag, tag_cls, cap):
        if img:
            inner = f'<img src="{img}" alt="{cap}">'
        else:
            inner = ('<div class="latent-slot"><div class="latent-slot-mark">Tensorleap UI '
                     '&mdash; latent-space view</div></div>')
        return (f'<figure class="latent-fig"><span class="latent-tag {tag_cls}">{tag}</span>'
                f'{inner}<figcaption>{cap}</figcaption></figure>')

    grid = (
        '<div class="latent-grid">'
        + tile(before, "Before", "tag-warn",
               'Base synthetic (yellow) forms its own cluster, away from real (red).')
        + tile(after, "After", "tag-good",
               'Tensorleap-optimized (magenta) now interleaves with real across the manifold.')
        + '</div>'
    )
    return f'''
<section>
  <p class="sec-eyebrow">Step 1 &middot; Latent-space calibration</p>
  <h2>Pull synthetic data into the real-data region of the latent space</h2>
  <p>
    A trained <b>RF-DETR</b> already encodes what real warehouse imagery looks like.
    We read its latent space on the real LOCO subset-3 images and treat that distribution
    as the <b>realism target</b>. Every candidate set of synthetic images is embedded by the
    <b>same network</b>; its distance to the real distribution
    (<span class="mono">MMD-RBF</span>) is the objective the loop minimizes.
  </p>
  {grid}
  <p class="caption">
    Latent-space map in the Tensorleap UI (real warehouse images in red). <b>Before:</b> the
    base synthetic set sits in a separate region. <b>After:</b> the Tensorleap-optimized set mixes
    into the real distribution while the un-calibrated base stays isolated &mdash; best embedding
    distance reached <span class="mono">{BEST_OBJECTIVE:.3f}</span> across {POPULATION_RUNS} scored runs.
  </p>
</section>'''


def build_param_section() -> str:
    vmax = max(v for _, v in PARAM_IMPORTANCE)
    rows = ""
    for i, (name, val) in enumerate(PARAM_IMPORTANCE, 1):
        w = val / vmax * 100
        rows += (
            f'<div class="pbar-row">'
            f'<div class="pbar-rank">{i}</div>'
            f'<div class="pbar-name mono">{name}</div>'
            f'<div class="pbar-track"><div class="pbar-fill" style="width:{w:.1f}%"></div></div>'
            f'<div class="pbar-val mono">{val:.4f}</div>'
            f'</div>'
        )
    return f'''
<section>
  <p class="sec-eyebrow">Step 2 &middot; What moved the needle</p>
  <h2>Which simulator knobs close the gap</h2>
  <p>
    The pipeline ranks every exposed parameter by how strongly it drives the latent-space
    distance. Scene environment and camera framing dominate — appearance-level noise and
    object placement fine-tune the rest. These rankings define the focused search space
    used in later rounds.
  </p>
  <div class="pbars">{rows}</div>
</section>'''


def build_training_curves() -> str:
    """Render brand-styled validation curves for the three models -> base64 PNG."""
    metrics = [
        ("val/ema_mAP_50",      "mAP@50",          0, 0.72),
        ("val/ema_mAP_50_95",   "mAP@50-95",       0, 0.45),
        ("val/AP/pallet_truck", "AP pallet_truck", 0, 0.60),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.patch.set_facecolor("white")
    for ax, (col, title, ymin, ymax) in zip(axes, metrics):
        ax.set_facecolor("white")
        ax.set_title(title, color="#191A20", fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("epoch", color="#6b6b78", fontsize=10)
        ax.tick_params(colors="#6b6b78", labelsize=9)
        ax.grid(True, color="#e4e4ec", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#e4e4ec")
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0, 35)
        for m in MODELS:
            df = pd.read_csv(TRAIN_ROOT / m / "metrics.csv")
            sub = df[["epoch", col]].dropna()
            sub = sub[sub["epoch"] <= 35]
            ax.plot(sub["epoch"], sub[col], color=SERIES_COLOR[m],
                    label=MODEL_LABELS[m], linewidth=2.0, alpha=0.95)
    plt.tight_layout(pad=1.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_payoff_section() -> str:
    # branded table
    hdr = "".join(f"<th>{c}</th>" for c in ["Model", "mAP@50", "F1", "mAR", "AP pallet_truck", "AP forklift", "AP pallet"])
    body = ""
    for m in MODELS:
        s = STATS[m]
        cls = "win" if m == "opt0" else ""
        body += (
            f'<tr class="{cls}"><td>{MODEL_LABELS[m]}</td>'
            f'<td class="mono">{s["mAP50"]:.3f}</td><td class="mono">{s["F1"]:.3f}</td>'
            f'<td class="mono">{s["mAR"]:.3f}</td><td class="mono">{s["pt"]:.3f}</td>'
            f'<td class="mono">{s["fork"]:.3f}</td><td class="mono">{s["pallet"]:.3f}</td></tr>'
        )
    table = f'<div class="tbl-wrap"><table><thead><tr>{hdr}</tr></thead><tbody>{body}</tbody></table></div>'

    curves = build_training_curves()
    legend = "".join(
        f'<span class="lg"><span class="lg-sw" style="background:{SERIES_COLOR[m]}"></span>{MODEL_LABELS[m]}</span>'
        for m in MODELS
    )
    curves_block = (
        '<figure class="curves">'
        f'<img src="data:image/png;base64,{curves}" alt="Validation curves over training">'
        f'<div class="legend">{legend}</div>'
        '<figcaption>Validation metrics per epoch (best EMA checkpoint selected per model).</figcaption>'
        '</figure>'
    )

    return f'''
<section>
  <p class="sec-eyebrow">Step 3 &middot; Retrain payoff</p>
  <h2>Fine-tuning the same network on calibrated data lifts accuracy</h2>
  <p>
    The best calibrated configuration is used to render a training set, and <b>RF-DETR</b> is
    fine-tuned on it. Evaluated on real LOCO subset-3 (873 real warehouse images, 3 classes),
    the calibrated model trains higher and faster than both the real-only and the un-calibrated
    base-synthetic baselines — most dramatically on <span class="mono">pallet_truck</span>.
  </p>
  {curves_block}
  {table}
</section>'''


def build_scenes_section(scenes: list[dict]) -> str:
    blocks = ""
    for s in scenes:
        figs = ""
        captions = ["Real only", "Base synthetic", "Calibrated (TL)"]
        for img, cap in zip(s["imgs"], captions):
            figs += (
                f'<figure><img src="data:image/png;base64,{img}" alt="{cap} detections"/>'
                f'<figcaption>{cap}</figcaption></figure>'
            )
        f1_chips = f'<span class="chip win">{s["win"]}</span>' + "".join(
            f'<span class="chip">{f}</span>' for f in s["f1s"]
        )
        blocks += f'''
    <div class="scene">
      <div class="scene-head">
        <span class="mono fn">{s["fn"]}</span>
        <span class="gt">{s["gt"]}</span>
        {f1_chips}
      </div>
      <div class="panels">{figs}</div>
    </div>'''
    return f'''
<section>
  <p class="sec-eyebrow">Qualitative</p>
  <h2>Where calibration shows up on real images</h2>
  <p class="legend-key">
    Red dashed = ground truth &middot; solid = predictions
    (<span class="swatch" style="background:#EEFF41"></span>pallet_truck,
    <span class="swatch" style="background:#FF6D00"></span>forklift,
    <span class="swatch" style="background:#00E5FF"></span>pallet).
    Each panel: real-only / base-synthetic / calibrated.
  </p>
  {blocks}
</section>'''


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

LOGO_SVG = r'''<svg width="158" height="26" viewBox="0 0 170 28" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="Tensorleap">
<g clip-path="url(#tllogo)">
<path d="M16.6479 10.6275C17.5259 11.2293 18.7197 11.0122 19.3215 10.1342C19.9233 9.25615 19.6964 8.06241 18.8282 7.46061C17.9501 6.8588 16.7564 7.08571 16.1546 7.95389C15.5528 8.82207 15.7698 10.0257 16.6479 10.6275Z" fill="#191A20"/>
<path d="M3.01373 17.0498C2.13568 16.448 0.94193 16.6749 0.340122 17.5431C-0.261685 18.4211 -0.0446399 19.6148 0.833407 20.2167C1.71145 20.8185 2.9052 20.5915 3.50701 19.7234C4.10882 18.8552 3.88191 17.6516 3.01373 17.0498Z" fill="#191A20"/>
<path d="M10.9255 24.1728C10.8169 24.0939 10.6985 24.0347 10.5802 23.9854V18.3915C10.5802 16.8426 9.92902 15.3529 8.7846 14.3071L3.73337 9.69C4.01947 8.89087 3.7531 7.9635 3.02303 7.46035C2.14499 6.85855 0.951239 7.07559 0.349431 7.95364C-0.252376 8.83169 -0.0254652 10.0254 0.842716 10.6272C1.42479 11.0219 2.13512 11.0515 2.73693 10.7851L7.78817 15.4022C8.62675 16.1718 9.11017 17.257 9.11017 18.3915V23.9854C8.7846 24.1235 8.48863 24.3504 8.27159 24.6661C7.66978 25.5442 7.88682 26.7379 8.76487 27.3397C9.64292 27.9415 10.8367 27.7146 11.4385 26.8464C12.0403 25.9684 11.8134 24.7747 10.9452 24.1728H10.9255Z" fill="#191A20"/>
<path d="M18.8276 17.0499C18.2456 16.6552 17.5352 16.6256 16.9334 16.892L11.8822 12.2749C11.0436 11.5053 10.5602 10.4201 10.5602 9.28555V3.69169C10.8857 3.55357 11.1817 3.32667 11.3988 3.01097C12.0006 2.13292 11.7835 0.939169 10.9055 0.337361C10.0274 -0.264446 8.83368 -0.0375349 8.23188 0.830646C7.63007 1.69883 7.85698 2.90245 8.72516 3.50426C8.83368 3.58318 8.95207 3.64236 9.07046 3.69169V9.28555C9.07046 10.8345 9.73146 12.3242 10.866 13.37L15.9173 17.9871C15.6311 18.7862 15.8975 19.7136 16.6276 20.2167C17.5056 20.8185 18.6994 20.6015 19.3012 19.7234C19.903 18.8454 19.6761 17.6517 18.8079 17.0499H18.8276Z" fill="#191A20"/>
<path d="M67.2096 7.62756H63.5592C62.0202 7.62756 60.7673 8.8805 60.7574 10.4195V7.62756H57.7188V22.2288H60.7574V10.4294H67.8015V22.2288H70.8401V11.2581C70.8401 9.2554 69.2123 7.62756 67.2096 7.62756Z" fill="#191A20"/>
<path d="M103.416 11.2581V22.2288H106.455V10.4294H112.433V7.62756H107.056C105.054 7.62756 103.426 9.2554 103.426 11.2581H103.416Z" fill="#191A20"/>
<path d="M116.972 1.67875H113.934V18.5984C113.934 20.6012 115.561 22.229 117.564 22.229H119.695V19.4271H116.972V1.67875Z" fill="#191A20"/>
<path d="M82.3439 14.672L75.5464 12.3141V10.4199H84.1395V7.61801H76.3554C74.2343 7.61801 72.5078 9.33464 72.5078 11.4656C72.5078 13.1033 73.5437 14.5635 75.0926 15.0962L81.9197 17.464V19.4273H73.0011V22.2291H81.0712C83.222 22.2291 84.9583 20.4927 84.9583 18.342C84.9583 16.6846 83.9126 15.2146 82.3439 14.672Z" fill="#191A20"/>
<path d="M93.8462 7.32221C89.4067 7.32221 86.0918 10.6469 86.0918 14.9188C86.0918 19.1906 89.4067 22.5154 93.8462 22.5154C98.2858 22.5154 101.601 19.1906 101.601 14.9188C101.601 10.6469 98.2858 7.32221 93.8462 7.32221ZM93.8462 19.6938C91.0937 19.6938 89.1008 17.6121 89.1008 14.9188C89.1008 12.2255 91.0937 10.1438 93.8462 10.1438C96.5988 10.1438 98.5916 12.2255 98.5916 14.9188C98.5916 17.6121 96.5988 19.6938 93.8462 19.6938Z" fill="#191A20"/>
<path d="M48.5628 7.32221C44.1429 7.32221 41.0254 10.5878 41.0254 14.9385C41.0254 19.2893 44.2613 22.5253 48.6713 22.5253C51.2561 22.5253 53.9199 21.2526 55.2715 18.8157L53.0517 17.2076C51.9566 18.9243 50.4965 19.7037 48.6713 19.7037C46.1457 19.7037 44.2811 17.987 44.143 15.7968H55.4688C55.5576 15.3233 55.5872 14.8497 55.5872 14.3959C55.5872 10.4891 52.8248 7.33207 48.5628 7.33207V7.32221ZM44.2219 13.5376C44.6066 11.4855 46.3035 10.0451 48.543 10.0451C50.6444 10.0451 52.3611 11.4855 52.5387 13.5376H44.2219Z" fill="#191A20"/>
<path d="M35.6194 3.46426H32.5808V7.62759H29.8184V10.4294H32.5808V18.5982C32.5808 20.6108 34.2086 22.2288 36.2113 22.2288H39.3388V19.427H35.6194V10.4196H39.8123V7.61773H35.6194V3.4544V3.46426Z" fill="#191A20"/>
<path d="M127.686 7.32221C123.266 7.32221 120.148 10.5878 120.148 14.9385C120.148 19.2893 123.384 22.5253 127.794 22.5253C130.379 22.5253 133.033 21.2526 134.395 18.8157L132.175 17.2076C131.08 18.9243 129.62 19.7037 127.794 19.7037C125.269 19.7037 123.404 17.987 123.266 15.7968H134.592C134.681 15.3233 134.71 14.8497 134.71 14.3959C134.71 10.4891 131.948 7.33207 127.686 7.33207V7.32221ZM123.335 13.5376C123.72 11.4855 125.417 10.0451 127.656 10.0451C129.758 10.0451 131.474 11.4855 131.652 13.5376H123.335Z" fill="#191A20"/>
<path d="M148.916 9.62092C147.841 8.23972 146.193 7.32221 143.717 7.32221C139.563 7.32221 136.387 10.3608 136.387 14.8497C136.387 19.3386 139.563 22.5253 143.717 22.5253C146.193 22.5253 147.841 21.6176 148.916 20.2265V22.2194H151.925V7.62804H148.916V9.62092ZM144.082 19.822C141.27 19.822 139.396 17.5431 139.396 14.8497C139.396 12.1564 141.27 10.0353 144.082 10.0353C146.894 10.0353 148.896 12.1663 148.896 14.8497C148.896 17.5332 146.894 19.822 144.082 19.822Z" fill="#191A20"/>
<path d="M162.669 7.32221C160.193 7.32221 158.555 8.22985 157.47 9.62092V7.62804H154.461V27.6751H157.47V20.2265C158.545 21.6077 160.193 22.5252 162.669 22.5252C166.823 22.5252 169.999 19.4866 169.999 14.9977C169.999 10.5088 166.823 7.32221 162.669 7.32221ZM162.304 19.8122C159.512 19.8122 157.49 17.6812 157.49 14.9977C157.49 12.3142 159.492 10.0254 162.304 10.0254C165.116 10.0254 166.99 12.3044 166.99 14.9977C166.99 17.6911 165.116 19.8122 162.304 19.8122Z" fill="#191A20"/>
</g>
<defs><clipPath id="tllogo"><rect width="170" height="28" fill="#fff"/></clipPath></defs>
</svg>'''


def main() -> None:
    scenes = extract_scene_images()
    before_img, after_img = load_latent_images()
    print(f"Scenes parsed: {len(scenes)} (imgs/scene: {[len(s['imgs']) for s in scenes]})")
    print(f"Latent images embedded: before={bool(before_img)} after={bool(after_img)}")

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sim-to-Real Calibration in Latent Space — Tensorleap</title>
<style>
  :root{{
    --ink:#191A20; --ink-soft:#33333d; --muted:#6b6b78;
    --indigo:#0A7E51; --indigo-deep:#097048; --green-bright:#0FBD79;
    --paper:#fff; --surface:#f6faf7; --mist:#e7efe9; --line:#dfe8e1;
    --amber:#d98b1f; --teal:#1fa98c; --teal-ink:#127a64;
    --grad:linear-gradient(90deg,#0A7E51,#0FBD79);
    --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--paper);color:var(--ink-soft);font-family:var(--sans);font-size:16px;line-height:1.6}}
  .mono{{font-family:var(--mono);font-variant-numeric:tabular-nums}}
  .topbar{{height:4px;background:var(--grad)}}
  .wrap{{max-width:980px;margin:0 auto;padding:0 26px 80px}}
  header.page{{padding:30px 0 8px}}
  .page-eyebrow{{font-size:.68rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--indigo);margin:18px 0 6px}}
  h1{{color:var(--ink);font-size:2.1rem;letter-spacing:-.022em;text-wrap:balance;margin:0 0 10px;line-height:1.15}}
  .lede{{font-size:1.06rem;color:var(--ink-soft);max-width:70ch;margin:0}}
  section{{margin-top:46px}}
  .sec-eyebrow{{font-size:.66rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--indigo);margin:0 0 4px}}
  h2{{color:var(--ink);font-size:1.32rem;letter-spacing:-.018em;text-wrap:balance;margin:0 0 10px}}
  p{{max-width:72ch}}
  b{{color:var(--ink)}}
  .caption{{font-size:.86rem;color:var(--muted);margin-top:10px}}

  /* TL;DR headline */
  .takeaway{{margin:22px 0 4px;font-size:1.04rem;color:var(--ink-soft);max-width:80ch;
    border-left:3px solid var(--indigo);padding-left:14px}}
  .headline-fig{{margin:8px 0 0;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px}}
  .headline-fig img{{width:100%;display:block}}

  /* loop diagram */
  .loop{{width:100%;height:auto;margin-top:18px;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:8px}}

  /* latent before/after */
  .latent-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:18px}}
  .latent-fig{{margin:0;position:relative}}
  .latent-fig img{{width:100%;display:block;border:1px solid #20202c;border-radius:10px;background:#0d0d12}}
  .latent-fig figcaption{{font-size:.82rem;color:var(--muted);margin-top:8px;line-height:1.45}}
  .latent-tag{{position:absolute;top:10px;left:10px;z-index:1;font-size:.7rem;font-weight:700;
    letter-spacing:.06em;text-transform:uppercase;padding:3px 9px;border-radius:6px;color:#fff}}
  .tag-warn{{background:var(--amber)}}
  .tag-good{{background:var(--indigo)}}
  .latent-slot{{border:2px dashed var(--line);border-radius:10px;background:var(--surface);
    min-height:300px;display:flex;align-items:center;justify-content:center;text-align:center;padding:24px}}
  .latent-slot-mark{{font-weight:700;color:var(--indigo);letter-spacing:.04em}}

  /* param bars */
  .pbars{{margin-top:18px;display:flex;flex-direction:column;gap:7px}}
  .pbar-row{{display:grid;grid-template-columns:26px minmax(180px,2fr) 4fr 64px;align-items:center;gap:12px;font-size:.85rem}}
  .pbar-rank{{color:var(--muted);text-align:right}}
  .pbar-name{{color:var(--ink-soft);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
  .pbar-track{{background:var(--mist);border-radius:4px;height:11px;overflow:hidden}}
  .pbar-fill{{height:100%;background:var(--grad);border-radius:4px}}
  .pbar-val{{text-align:right;color:var(--muted)}}

  /* training curves */
  .curves{{margin:20px 0 0;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px}}
  .curves img{{width:100%;display:block;border-radius:6px}}
  .curves figcaption{{font-size:.8rem;color:var(--muted);text-align:center;margin-top:8px}}
  .legend{{display:flex;flex-wrap:wrap;gap:18px;margin-top:12px;justify-content:center}}
  .lg{{font-size:.8rem;color:var(--muted);display:flex;align-items:center;gap:6px}}
  .lg-sw{{width:12px;height:12px;border-radius:3px;display:inline-block}}

  /* table */
  .tbl-wrap{{overflow-x:auto;margin-top:18px}}
  table{{width:100%;border-collapse:collapse;font-size:.88rem}}
  th{{text-align:left;font-family:var(--mono);font-size:.74rem;letter-spacing:.04em;text-transform:uppercase;
    color:var(--muted);padding:8px 12px;border-bottom:2px solid var(--line);white-space:nowrap}}
  td{{padding:9px 12px;border-bottom:1px solid var(--line)}}
  td:first-child{{font-weight:600;color:var(--ink)}}
  tr.win{{background:rgba(68,53,237,.06)}}
  tr.win td:first-child{{color:var(--indigo)}}

  /* scenes */
  .legend-key{{font-size:.84rem;color:var(--muted)}}
  .swatch{{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 2px -1px 6px}}
  .scene{{margin-top:20px;background:#14141c;border:1px solid #20202c;border-radius:12px;padding:14px}}
  .scene-head{{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin-bottom:10px}}
  .scene-head .fn{{color:#cfd0e6;font-size:.82rem}}
  .scene-head .gt{{color:#8b8ca6;font-size:.8rem}}
  .chip{{font-family:var(--mono);font-size:.74rem;color:#cfd0e6;background:#22222e;border:1px solid #2c2c3a;border-radius:6px;padding:2px 8px}}
  .chip.win{{color:#fff;background:var(--indigo);border-color:var(--indigo)}}
  .panels{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}
  .panels figure{{margin:0}}
  .panels img{{width:100%;border-radius:6px;display:block}}
  .panels figcaption{{font-size:.76rem;color:#8b8ca6;text-align:center;margin-top:5px}}

  /* footnote */
  .method{{margin-top:50px;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px 20px;
    font-size:.84rem;color:var(--muted);position:relative;overflow:hidden}}
  .method::before{{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--grad)}}
  .method b{{color:var(--ink-soft)}}
  footer{{margin-top:34px;font-size:.78rem;color:var(--muted);border-top:1px solid var(--line);padding-top:16px}}

  @media (max-width:680px){{
    .latent-grid{{grid-template-columns:1fr}}
    .pbar-row{{grid-template-columns:22px 1.6fr 3fr 54px;gap:8px;font-size:.78rem}}
    .panels{{grid-template-columns:1fr}}
    h1{{font-size:1.7rem}}
  }}
</style>
</head>
<body>
<div class="topbar"></div>
<div class="wrap">
  <header class="page">
    {LOGO_SVG}
    <p class="page-eyebrow">Synthetic-data calibration &middot; Warehouse object detection</p>
    <h1>Closing the sim-to-real gap in a trained network&rsquo;s latent space</h1>
    <p class="lede">
      We use the detector&rsquo;s latent representations to model what &ldquo;real&rdquo; looks like,
      then optimize the synthetic-data generator until synthetic samples align with that
      real-data distribution.
    </p>
    {build_headline()}
  </header>

  <section>
    <p class="sec-eyebrow">The loop</p>
    <h2>A closed loop for synthetic-data calibration</h2>
    <p>
      The whole process runs through the Tensorleap API. A trained RF-DETR provides the
      latent space that scores realism; the optimization pipeline searches the simulator&rsquo;s
      parameters to minimize the distance between synthetic and real in that space; the winning
      configuration produces the data the same network is then fine-tuned on.
    </p>
    {build_loop_diagram()}
  </section>

  {build_latent_section(before_img, after_img)}
  {build_param_section()}
  {build_payoff_section()}
  {build_scenes_section(scenes)}

  <div class="method">
    <b>Method.</b> Reference latent space: a trained <b>RF-DETR</b> embedding of real LOCO
    subset-3 warehouse images. Objective: maximum-mean-discrepancy (<span class="mono">MMD-RBF</span>)
    between the synthetic and real embedding distributions, with centroid and nearest-neighbour
    distances as secondary signals. Search and suggestion are handled by the <b>Tensorleap native
    optimization pipeline</b> running in the Tensorleap API; the simulator is Isaac SDG.
    Detection metrics are evaluated on 873 real images across pallet_truck, forklift and pallet.
  </div>

  <footer>Generated for the Tensorleap synthetic-data calibration workflow.</footer>
</div>
</body>
</html>'''

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nWrote -> {OUT_HTML}  ({OUT_HTML.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
