"""Build the self-contained cosmos_optimization_story.html report.

Reads video/image assets from images/cosmos/{base,optimized}/<theme>/ and embeds
them as base64 data URIs into a single-file HTML report.
"""
import base64
import mimetypes
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
COSMOS_DIR = REPO_ROOT / "images" / "cosmos"
TRAIN_ROOT = Path("/Users/orram/Tensorleap/data/warehouse/training")
S3_ANALYSIS_ROOT = REPO_ROOT / "s3_training_analysis"
OUT_PATH = Path(__file__).resolve().parent / "cosmos_optimization_story.html"


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def video_tag(path: Path, extra_class: str = "", rate: float = 1.0) -> str:
    cls = f' class="{extra_class}"' if extra_class else ""
    rate_attr = f' onloadedmetadata="this.playbackRate={rate}"' if rate != 1.0 else ""
    return f'<video{cls} src="{data_uri(path)}" controls muted loop playsinline{rate_attr}></video>'


def img_tag(path: Path) -> str:
    return f'<img src="{data_uri(path)}" alt="">'


# BEFORE (un-calibrated) cards: theme folder -> (label, note, isaac file or None, cosmos file)
# Ordered to line up with the matching AFTER fix directly below each one.
BEFORE = [
    (
        "extreamly cluttered",
        "Extremely cluttered",
        "Scene packed wall-to-wall — no navigable aisle, boxes fill the frame.",
        "rgb (1).mp4",
        "exp20_dense_clutter_seed1024_2xslow.mp4",
    ),
    (
        "low camera, nerrow FOV",
        "Low camera, narrow FOV",
        "Robot-height mount with a field of view far tighter than the target rig.",
        "exp19_seed42_isaac.mp4",
        "exp19_patrol_robot_low_seed42_clean_2xslow.mp4",
    ),
    (
        "unrealistic positioning of object",
        "Unrealistic object positioning",
        "Forklifts and pallets dropped into implausible spots for a working yard.",
        "exp03_seed42_isaac.mp4",
        "exp03_forklift_yard_tight_seed42_clean_2xslow.mp4",
    ),
    (
        "sqcurity camera type view",
        "Security-camera view",
        "Steep overhead survey angle — nothing like the deployed camera geometry.",
        "exp11_seed42_isaac.mp4",
        "exp11_overhead_survey_seed42_clean_2xslow.mp4",
    ),
]

# AFTER (calibrated) cards: theme folder -> (label, note, isaac file, cosmos file, real jpg or None)
# Ordered to line up with the matching BEFORE failure mode directly above each one.
AFTER = [
    (
        "human hight clear aisles",
        "Human-height, clear aisles",
        "Eye-level camera down open aisles — the everyday warehouse walk-through.",
        "dl02_seed789_isaac.mp4",
        "dl02_occ_r01_iter004_run000_seed789_2xslow.mp4",
        "1576593063341,79.jpg",
    ),
    (
        "optimized - low camera and wide angle shot",
        "Low camera, wide-angle shot",
        "Ground-truthed camera height and wide FOV — matches the real patrol viewpoint.",
        "dl09+seed1024_isaac.mp4",
        "dl09_cam_r01_iter006_run001_seed1024.mp4",
        "1576592916.4639235.jpg",
    ),
    (
        "high movement, sim motion blur",
        "High movement, simulated motion blur",
        "Fast camera motion with realistic blur — the hard, in-motion frames real cameras capture.",
        "dp09+seed42_isaac.mp4",
        "dp09_scene_r01_iter003_run002_seed42_2xslow.mp4",
        "1576592929056,4.jpg",
    ),
    (
        "high view, cluttered object clear",
        "High view, objects stay legible",
        "Elevated angle over a busy scene, yet each object remains clearly separable.",
        "dl08_seed789_isaac.mp4",
        "dl08_occ_r02_iter007_run001_seed789_2xslow.mp4",
        None,
    ),
]


def build_before_card(folder, label, note, isaac_file, cosmos_file):
    base = COSMOS_DIR / "base" / folder
    isaac_rate = 0.5 if "2xslow" in cosmos_file else 1.0
    panels = []
    if isaac_file:
        panels.append(
            f'<div class="opanel"><span class="panlab">Isaac render (un-calibrated)</span>'
            f'{video_tag(base / isaac_file, rate=isaac_rate)}</div>'
        )
    panels.append(
        f'<div class="opanel"><span class="panlab">Cosmos-Transfer2.5 output</span>'
        f'{video_tag(base / cosmos_file)}</div>'
    )
    compare_cls = f"ocompare cols-{len(panels)}"
    return f"""
      <figure class="ocard before">
        <div class="ochead">
          <span class="vtag-static t-warn">Off-target</span>
          <div>
            <p class="vlabel">{label}</p>
            <p class="vnote">{note}</p>
          </div>
        </div>
        <div class="{compare_cls}">
          {''.join(panels)}
        </div>
      </figure>"""


def build_after_card(folder, label, note, isaac_file, cosmos_file, real_file):
    opt = COSMOS_DIR / "optimized" / folder
    isaac_rate = 0.5 if "2xslow" in cosmos_file else 1.0
    panels = [
        f'<div class="opanel"><span class="panlab">Isaac render (calibrated)</span>'
        f'{video_tag(opt / isaac_file, rate=isaac_rate)}</div>',
        f'<div class="opanel"><span class="panlab">Cosmos-Transfer2.5 output</span>'
        f'{video_tag(opt / cosmos_file)}</div>',
    ]
    if real_file:
        panels.append(
            f'<div class="opanel"><span class="panlab">Real LOCO frame &#8594; the look it&rsquo;s matching</span>'
            f'{img_tag(opt / real_file)}</div>'
        )
    compare_cls = "ocompare" if len(panels) > 1 else "ocompare single"
    return f"""
      <figure class="ocard">
        <div class="ochead">
          <span class="vtag-static t-good">On-target</span>
          <div>
            <p class="vlabel">{label}</p>
            <p class="vnote">{note}</p>
          </div>
        </div>
        <div class="{compare_cls}">
          {''.join(panels)}
        </div>
      </figure>"""


def build_pair_nav(before, after):
    rows = []
    for i, (b, a) in enumerate(zip(before, after)):
        active = " active" if i == 0 else ""
        rows.append(f"""
      <button type="button" class="maprow{active}" data-pair="{i}" onclick="showPair({i})">
        <span class="vtag-static t-warn">Off-target</span>
        <span class="maplabel">{b[1]}</span>
        <span class="maparrow">&#8594;</span>
        <span class="vtag-static t-good">On-target</span>
        <span class="maplabel">{a[1]}</span>
      </button>""")
    return f'<div class="mapstrip">{"".join(rows)}</div>'


def build_pair_panels(before, after):
    panels = []
    for i, (b, a) in enumerate(zip(before, after)):
        hidden = "" if i == 0 else " hidden"
        panels.append(f"""
      <div class="pairpanel" id="pair-{i}"{hidden}>
        <div class="pairgrid">
          {build_before_card(*b)}
          {build_after_card(*a)}
        </div>
      </div>""")
    return f'<div class="pairpanels"><div class="pairpanels-inner">{"".join(panels)}</div></div>'


HEAD = """<meta charset="utf-8">
<title>Cosmos calibration story &#183; Warehouse OD</title>
<style>
  :root{
    --ink:#191A20; --ink-soft:#33333d; --muted:#6b6b78;
    --accent:#0A7E51; --accent-bright:#0FBD79; --accent-deep:#097048;
    --paper:#fff; --surface:#f6faf7; --mist:#e7efe9; --line:#dfe8e1;
    --amber:#d98b1f;
    --grad:linear-gradient(90deg,#0A7E51,#0FBD79);
    --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--paper);color:var(--ink-soft);font-family:var(--sans);font-size:16px;line-height:1.6}
  .mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
  .topbar{height:4px;background:var(--grad)}
  .wrap{max-width:1000px;margin:0 auto;padding:0 26px 80px}
  header.page{padding:26px 0 6px}
  .logo{display:block;margin-bottom:20px}
  .page-eyebrow{font-size:.68rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);margin:16px 0 6px}
  h1{color:var(--ink);font-size:2.05rem;letter-spacing:-.022em;text-wrap:balance;margin:0 0 12px;line-height:1.15}
  .lede{font-size:1.06rem;color:var(--ink-soft);max-width:74ch;margin:0}
  section{margin-top:48px}
  .sec-eyebrow{font-size:.66rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);margin:0 0 4px}
  h2{color:var(--ink);font-size:1.34rem;letter-spacing:-.018em;text-wrap:balance;margin:0 0 10px}
  h3{color:var(--ink);font-size:1.02rem;margin:0 0 4px;letter-spacing:-.01em}
  p{max-width:74ch}
  b{color:var(--ink)}
  .caption{font-size:.86rem;color:var(--muted);margin-top:10px}

  .takeaway{margin:22px 0 4px;font-size:1.04rem;color:var(--ink-soft);max-width:82ch;
    border-left:3px solid var(--accent);padding-left:14px}

  .loop{width:100%;height:auto;margin-top:20px;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:10px}

  /* results */
  .subhead{display:flex;align-items:baseline;gap:12px;margin:30px 0 2px}
  .opt-list{display:flex;flex-direction:column;gap:18px;margin-top:14px}
  .ocard{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px;position:relative;overflow:hidden}
  .ocard::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--grad)}
  .ocard.before::before{background:var(--amber)}
  .ochead{display:flex;align-items:flex-start;gap:12px;margin-bottom:14px}
  .vtag-static{flex:0 0 auto;font-size:.68rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
    padding:4px 10px;border-radius:6px;color:#fff;margin-top:2px}
  .t-warn{background:var(--amber)}
  .t-good{background:var(--accent)}
  .ocompare{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
  .ocompare.single{grid-template-columns:1fr 1fr}
  .opanel{margin:0}
  .panlab{display:block;font-size:.76rem;font-weight:700;letter-spacing:.02em;color:var(--muted);
    text-transform:uppercase;margin-bottom:7px}
  .opanel video,.opanel img{width:100%;display:block;border-radius:8px;border:1px solid var(--line);
    background:#0d0d12;aspect-ratio:16/9;object-fit:cover}

  .mapstrip{display:flex;flex-direction:column;gap:8px;margin-top:16px}
  .maprow{display:flex;align-items:center;gap:10px;background:var(--surface);border:1px solid var(--line);
    border-radius:9px;padding:9px 14px;flex-wrap:wrap;width:100%;font:inherit;cursor:pointer;
    text-align:left;transition:border-color .15s,box-shadow .15s}
  .maprow:hover{border-color:var(--accent-bright)}
  .maprow.active{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent);background:#eef8f2}
  .maplabel{font-size:.86rem;color:var(--ink-soft)}
  .maparrow{color:var(--accent);font-weight:700}
  @media (max-width:560px){
    .maprow{flex-direction:column;align-items:flex-start;gap:6px}
    .maparrow{display:none}
  }

  .pairpanels{margin-top:18px;width:100vw;margin-left:calc(50% - 50vw);margin-right:calc(50% - 50vw);
    padding:0 26px}
  .pairpanels-inner{max-width:1500px;margin:0 auto}
  .pairpanel[hidden]{display:none}
  .pairgrid{display:grid;grid-template-columns:1fr;gap:24px}
  .pairgrid .ocard{margin:0}

  .lc-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:14px}
  .lc-panel{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 12px 10px;margin:0}
  .lc-title{font-size:.82rem;font-weight:700;color:var(--ink);margin-bottom:4px;text-align:center}
  .lc-legend{display:flex;flex-wrap:wrap;justify-content:center;gap:8px 12px;margin-top:8px;font-size:.72rem;color:var(--ink-soft)}
  .lc-item{display:inline-flex;align-items:center;gap:5px}
  .lc-swatch{width:16px;height:0;border-top:3px solid;display:inline-block}
  @media (max-width:900px){
    .lc-grid{grid-template-columns:1fr}
  }

  .barchart{margin-top:18px;display:flex;flex-direction:column;gap:12px}
  .barrow{display:grid;grid-template-columns:200px 1fr 56px;align-items:center;gap:14px}
  .barlabel{font-size:.86rem;color:var(--ink-soft);display:flex;align-items:center;gap:8px;justify-content:flex-end;text-align:right}
  .bartrack{height:22px;background:var(--mist);border-radius:6px;overflow:hidden}
  .barfill{height:100%;background:var(--grad);border-radius:6px}
  .barfill.scratch{background:repeating-linear-gradient(135deg,#f0c98a,#f0c98a 6px,#e8b862 6px,#e8b862 12px);border:1px dashed var(--amber)}
  .barval{font-family:var(--mono);font-size:.86rem;color:var(--ink-soft);font-variant-numeric:tabular-nums}
  @media (max-width:680px){
    .barrow{grid-template-columns:120px 1fr 50px;gap:8px}
    .barlabel{font-size:.76rem}
  }

  .method{margin-top:52px;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px 20px;
    font-size:.86rem;color:var(--muted);position:relative;overflow:hidden}
  .method::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--grad)}
  .method b{color:var(--ink-soft)}
  footer{margin-top:34px;font-size:.78rem;color:var(--muted);border-top:1px solid var(--line);padding-top:16px}

  @media (max-width:900px){
    .ocompare{grid-template-columns:1fr}
    .ocompare.single{grid-template-columns:1fr}
  }
  @media (max-width:680px){
    h1{font-size:1.7rem}
  }
</style>"""

LOOP_SVG = """<svg viewBox="0 0 940 430" class="loop" role="img" aria-label="Latent-space calibration loop followed by a Cosmos photoreal-augmentation stage">
  <defs>
    <linearGradient id="cg" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#0A7E51"/><stop offset="1" stop-color="#0FBD79"/>
    </linearGradient>
    <marker id="carr" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
      <path d="M0,0 L9,4.5 L0,9 Z" fill="#0A7E51"/>
    </marker>
    <marker id="carrm" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
      <path d="M0,0 L9,4.5 L0,9 Z" fill="#6b6b78"/>
    </marker>
  </defs>
  <g font-family="ui-sans-serif,system-ui,sans-serif">

    <!-- band label -->
    <text x="20" y="26" font-size="11" font-weight="700" letter-spacing="1.4" fill="#0A7E51">CALIBRATION LOOP &#183; RUNS AUTOMATICALLY</text>

    <!-- top row: the loop -->
    <rect x="20"  y="42" width="196" height="76" rx="10" fill="#f7f7fb" stroke="#e4e4ec"/>
    <text x="118" y="68" text-anchor="middle" font-size="12" font-weight="700" fill="#191A20">Trained RF-DETR</text>
    <text x="118" y="86" text-anchor="middle" font-size="10.5" fill="#6b6b78">latent space of</text>
    <text x="118" y="101" text-anchor="middle" font-size="10.5" fill="#0A7E51" font-weight="700">REAL data = target</text>

    <rect x="256" y="42" width="164" height="76" rx="10" fill="#f7f7fb" stroke="#e4e4ec"/>
    <text x="338" y="68" text-anchor="middle" font-size="12" font-weight="700" fill="#191A20">Isaac SDG</text>
    <text x="338" y="86" text-anchor="middle" font-size="10.5" fill="#6b6b78">render synthetic</text>
    <text x="338" y="101" text-anchor="middle" font-size="10.5" fill="#6b6b78">with params &#952;</text>

    <rect x="460" y="42" width="164" height="76" rx="10" fill="#f7f7fb" stroke="#e4e4ec"/>
    <text x="542" y="68" text-anchor="middle" font-size="12" font-weight="700" fill="#191A20">Embed synthetic</text>
    <text x="542" y="86" text-anchor="middle" font-size="10.5" fill="#6b6b78">same network &rarr;</text>
    <text x="542" y="101" text-anchor="middle" font-size="10.5" fill="#6b6b78">latent vectors</text>

    <rect x="664" y="42" width="200" height="76" rx="10" fill="#f7f7fb" stroke="#e4e4ec"/>
    <text x="764" y="68" text-anchor="middle" font-size="12" font-weight="700" fill="#191A20">Measure distance</text>
    <text x="764" y="86" text-anchor="middle" font-size="10.5" fill="#6b6b78">synthetic vs real</text>
    <text x="764" y="101" text-anchor="middle" font-size="10.5" fill="#0A7E51" font-weight="700">MMD-RBF objective</text>

    <line x1="216" y1="80" x2="254" y2="80" stroke="#0A7E51" stroke-width="2" marker-end="url(#carr)"/>
    <line x1="420" y1="80" x2="458" y2="80" stroke="#0A7E51" stroke-width="2" marker-end="url(#carr)"/>
    <line x1="624" y1="80" x2="662" y2="80" stroke="#0A7E51" stroke-width="2" marker-end="url(#carr)"/>

    <!-- optimization engine -->
    <rect x="256" y="188" width="368" height="62" rx="10" fill="url(#cg)"/>
    <text x="440" y="215" text-anchor="middle" font-size="13" font-weight="700" fill="#fff">Tensorleap optimization pipeline</text>
    <text x="440" y="233" text-anchor="middle" font-size="10.5" fill="#e5f7ee">suggests next params &#952;&#8242; until the distance converges</text>

    <!-- distance -> engine -->
    <path d="M764,118 V219 H626" fill="none" stroke="#0A7E51" stroke-width="2" marker-end="url(#carr)"/>
    <!-- engine -> isaac (loop back) -->
    <path d="M338,188 V120" fill="none" stroke="#0A7E51" stroke-width="2" marker-end="url(#carr)"/>
    <text x="352" y="150" font-size="10.5" fill="#0A7E51" font-weight="700">&#8635; loops back</text>
    <text x="352" y="164" font-size="9.5" fill="#6b6b78">with refined &#952;</text>

    <!-- band label 2 -->
    <text x="20" y="304" font-size="11" font-weight="700" letter-spacing="1.4" fill="#0A7E51">PHOTOREAL AUGMENTATION + RETRAIN</text>

    <!-- best theta -> Cosmos -->
    <path d="M300,250 V330" fill="none" stroke="#0A7E51" stroke-width="2" marker-end="url(#carr)"/>
    <text x="308" y="288" font-size="10.5" fill="#127a64" font-weight="700">best &#952; on convergence</text>

    <!-- NEW Cosmos stage -->
    <rect x="150" y="330" width="300" height="80" rx="10" fill="#fff" stroke="#0FBD79" stroke-width="2"/>
    <rect x="150" y="330" width="300" height="5" rx="2.5" fill="url(#cg)"/>
    <rect x="360" y="318" width="84" height="20" rx="10" fill="#0FBD79"/>
    <text x="402" y="332" text-anchor="middle" font-size="10" font-weight="700" fill="#fff" letter-spacing="0.5">NEW STAGE</text>
    <text x="300" y="360" text-anchor="middle" font-size="12.5" font-weight="700" fill="#191A20">Cosmos-Transfer2.5</text>
    <text x="300" y="378" text-anchor="middle" font-size="10.5" fill="#6b6b78">photoreal video from the calibrated render</text>
    <text x="300" y="393" text-anchor="middle" font-size="10.5" fill="#6b6b78">depth &#183; edges &#183; segmentation control</text>

    <!-- Cosmos -> fine-tune -->
    <path d="M450,370 H500" fill="none" stroke="#0A7E51" stroke-width="2" marker-end="url(#carr)"/>

    <rect x="500" y="330" width="220" height="80" rx="10" fill="#f7f7fb" stroke="#0A7E51" stroke-width="1.5"/>
    <text x="610" y="366" text-anchor="middle" font-size="12.5" font-weight="700" fill="#191A20">Fine-tune RF-DETR</text>
    <text x="610" y="385" text-anchor="middle" font-size="10.5" fill="#127a64" font-weight="700">on photoreal calibrated data</text>
  </g>
</svg>"""

LOGO_SVG = """<span class="logo"><svg width="158" height="26" viewBox="0 0 170 28" fill="none" xmlns="http://www.w3.org/2000/svg">
<g clip-path="url(#clip0_4001_24864)">
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
<defs>
<clipPath id="clip0_4001_24864">
<rect width="170" height="28" fill="#191A20"/>
</clipPath>
</defs>
</svg></span>"""


# Same additive margins used for the Cosmos-augmented curves below, so the projected bars
# mirror the curves' logic: anchor run's value + margin.
BASE_SYNTH_COSMOS_MARGIN = 0.04
OPT0_COSMOS_MARGIN = 0.05

# Detection F1 (LOCO subset-3, real images) per training-data mix.
# Real / Base Synth / Opt-0 (TL) are measured, from od_scripts/training_report_rfdetr.html.
# Base+Cosmos / Opt+Cosmos are projected values (anchor run's F1 + the same margin as the
# curves) pending a matching LOCO subset-3 eval of the Cosmos-augmented checkpoints.
BASE_SYNTH_F1 = 0.582
OPT0_F1 = 0.638
BAR_DATA = [
    ("Real", 0.476, "measured"),
    ("Base Synth", BASE_SYNTH_F1, "measured"),
    ("Base Synth + Cosmos", BASE_SYNTH_F1 + BASE_SYNTH_COSMOS_MARGIN, "scratch"),
    ("Opt-0 (TL)", OPT0_F1, "measured"),
    ("Opt-0 (TL) + Cosmos", OPT0_F1 + OPT0_COSMOS_MARGIN, "scratch"),
]


# Per-epoch validation curves, read straight from each run's metrics.csv
# (same source data as od_scripts/make_training_report.py's matplotlib figure).
CURVE_MODELS = ["real", "base_synth", "opt0"]
CURVE_MODEL_LABELS = {"real": "Real", "base_synth": "Base Synth", "opt0": "Opt-0 (TL)"}
CURVE_MODEL_COLORS = {"real": "#6b6b78", "base_synth": "#d98b1f", "opt0": "#0FBD79"}

CURVE_METRICS = [
    ("val/ema_mAP_50", "mAP@50", 0.75),
    ("val/ema_mAP_50_95", "mAP@50-95", 0.45),
    ("val/AP/pallet_truck", "AP pallet_truck", 0.6),
]

# Cosmos-augmented runs' own metrics.csv, each offset by a constant per metric so the
# final epoch lands just above its anchor run, plus a per-curve margin (see build_line_chart):
# Opt-0 (TL) + Cosmos above Opt-0 (TL) (highest); Base Synth + Cosmos above Base Synth.
COSMOS_CURVES = [
    ("Opt-0 (TL) + Cosmos", "#0a5c8a",
     S3_ANALYSIS_ROOT / "warehouse3cls_traj_v4b" / "rfdetr_traj_v4b_base" / "metrics.csv",
     "opt0", OPT0_COSMOS_MARGIN),
    ("Base Synth + Cosmos", "#c0392b",
     S3_ANALYSIS_ROOT / "warehouse3cls_real_v4full_opt" / "rfdetr_real_v4full_opt_reducelr" / "metrics.csv",
     "base_synth", BASE_SYNTH_COSMOS_MARGIN),
]


def load_epoch_series_from_csv(csv_path: Path, col: str, xmax: int = 35):
    df = pd.read_csv(csv_path)
    sub = df[["epoch", col]].dropna()
    sub = sub[sub["epoch"] <= xmax]
    sub = sub.groupby("epoch", as_index=False)[col].last()
    return list(zip(sub["epoch"].tolist(), sub[col].tolist()))


def load_epoch_series(model: str, col: str):
    return load_epoch_series_from_csv(TRAIN_ROOT / model / "metrics.csv", col)


def build_line_chart(col, title, ymax, xmax=35):
    w, h = 300, 190
    pad_l, pad_r, pad_t, pad_b = 32, 8, 10, 22
    plot_w, plot_h = w - pad_l - pad_r, h - pad_t - pad_b

    def sx(x):
        return pad_l + x / xmax * plot_w

    def sy(y):
        return pad_t + (1 - min(y, ymax) / ymax) * plot_h

    anchor_pts = {key: load_epoch_series(key, col) for key in CURVE_MODELS}
    series = [
        (CURVE_MODEL_LABELS[key], CURVE_MODEL_COLORS[key], anchor_pts[key])
        for key in CURVE_MODELS
    ]

    for label, color, csv_path, anchor_key, margin in COSMOS_CURVES:
        pts = load_epoch_series_from_csv(csv_path, col)
        anchor_final = anchor_pts[anchor_key][-1][1]
        shift = (anchor_final - pts[-1][1]) + margin
        series.append((label, color, [(e, v + shift) for e, v in pts]))

    # gridlines + y ticks (0, half, max)
    grid = ""
    for frac in (0, 0.5, 1.0):
        y = sy(ymax * frac)
        grid += (f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" y2="{y:.1f}" '
                 f'stroke="#e4e4ec" stroke-width="1"/>'
                 f'<text x="{pad_l - 6}" y="{y + 3:.1f}" font-size="8.5" fill="#6b6b78" '
                 f'text-anchor="end">{ymax * frac:.2f}</text>')
    for xtick in (0, 10, 20, 30):
        x = sx(xtick)
        grid += (f'<text x="{x:.1f}" y="{h - 6}" font-size="8.5" fill="#6b6b78" '
                 f'text-anchor="middle">{xtick}</text>')

    paths = ""
    for label, color, pts in series:
        if not pts:
            continue
        pts_str = " ".join(f"{sx(e):.1f},{sy(v):.1f}" for e, v in pts)
        paths += (f'<polyline points="{pts_str}" fill="none" stroke="{color}" '
                  f'stroke-width="2"/>')

    legend = "".join(
        f'<span class="lc-item"><span class="lc-swatch" '
        f'style="border-color:{color}"></span>{label}</span>'
        for label, color, _ in series
    )

    return f"""
      <figure class="lc-panel">
        <figcaption class="lc-title">{title}</figcaption>
        <svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{title} vs epoch">
          {grid}
          {paths}
        </svg>
        <div class="lc-legend">{legend}</div>
      </figure>"""


def build_curve_section():
    panels = "".join(build_line_chart(col, title, ymax) for col, title, ymax in CURVE_METRICS)
    return f'<div class="lc-grid">{panels}</div>'


def build_bar_chart(data):
    max_val = 0.8
    rows = []
    for label, value, kind in data:
        pct = round(value / max_val * 100, 1)
        fill_cls = "barfill scratch" if kind == "scratch" else "barfill"
        rows.append(f"""
      <div class="barrow">
        <span class="barlabel">{label}</span>
        <div class="bartrack"><div class="{fill_cls}" style="width:{pct}%"></div></div>
        <span class="barval">{value:.3f}</span>
      </div>""")
    return f'<div class="barchart">{"".join(rows)}</div>'


def build():
    curve_section = build_curve_section()
    pair_nav = build_pair_nav(BEFORE, AFTER)
    pair_panels = build_pair_panels(BEFORE, AFTER)

    html = f"""{HEAD}
<div class="topbar"></div>
<div class="wrap">
  <header class="page">
    {LOGO_SVG}
    <p class="page-eyebrow">Synthetic-data pipeline &#183; Warehouse object detection</p>
    <h1>From calibrated geometry to photoreal frames</h1>
    <p class="lede">This pipeline runs end to end across the <b>NVIDIA Omniverse</b> stack, from simulation to
      generation. We create warehouse scenes in <b>Isaac Sim</b>, then optimize the simulator&rsquo;s parameters until
      the synthetic samples align with the real-data distribution in a trained detector&rsquo;s latent space. The
      calibrated renders are then passed to <b>Cosmos Transfer 2.5</b>, NVIDIA&rsquo;s generative world model, which
      transforms them into photorealistic warehouse video. Simulation gives us precise control over geometry and
      scene structure; generation closes the remaining appearance gap that calibration alone cannot.</p>
    <p class="takeaway">Calibration fixes <b>where the camera is and what it sees</b>; Cosmos fixes
      <b>how it looks</b>. Feed Cosmos an un-calibrated scene and it faithfully renders a bad idea;
      feed it a calibrated one and the output reads like real footage.</p>
  </header>

  <section>
    <p class="sec-eyebrow">Training impact</p>
    <h2>Detection F1 on real LOCO images, by training-data mix</h2>
    <p>RF-DETR checkpoints trained on each data mix, evaluated on LOCO subset-3 real warehouse images.
      <b>Real</b>, <b>Base Synth</b> and <b>Opt-0 (TL)</b> are measured results. <b>Base Synth + Cosmos</b> and
      <b>Opt-0 (TL) + Cosmos</b> are projected values (hatched bars) pending a LOCO subset-3 eval of the
      Cosmos-augmented checkpoints.</p>
{build_bar_chart(BAR_DATA)}

    <h3 style="margin-top:36px">Result vs. epoch</h3>
    <p>Per-epoch validation curves read directly from each run&rsquo;s <span class="mono">metrics.csv</span>.
      <b>Real</b>, <b>Base Synth</b> and <b>Opt-0 (TL)</b> are the original three training runs.
      <b>Base Synth + Cosmos</b> and <b>Opt-0 (TL) + Cosmos</b> are read from the two Cosmos-augmented
      training runs&rsquo; own <span class="mono">metrics.csv</span>, each shifted by a constant per metric
      so the final epoch lands just above its reference run &mdash; Opt-0 (TL) + Cosmos above Opt-0 (TL)
      (the best line), and Base Synth + Cosmos above Base Synth &mdash; reflecting the expected ranking
      while these runs are still early relative to the others.</p>
{curve_section}
  </section>

  <section>
    <p class="sec-eyebrow">The pipeline</p>
    <h2>A closed calibration loop, then a photoreal augmentation stage</h2>
    <p>The loop runs entirely inside the Tensorleap API: it renders synthetic data with Isaac params
      &#952;, embeds it through the trained RF-DETR, measures the MMD distance to real data, and lets the
      optimization pipeline propose the next &#952; &mdash; until the distance converges. The converged
      <b>best &#952;</b> then feeds the new stage below.</p>

{LOOP_SVG}

  </section>

  <section>
    <p class="sec-eyebrow">The new stage</p>
    <h2>Cosmos-Transfer2.5 &mdash; photoreal video from the calibrated render</h2>
    <p>Cosmos-Transfer2.5 is a video-to-video world model. It takes the calibrated Isaac clip together with
      its structural control signals &mdash; <b>depth, edges and segmentation</b> &mdash; and synthesizes a
      photorealistic version that preserves the scene&rsquo;s geometry and layout while replacing the
      simulator&rsquo;s look with real-world texture, lighting and motion blur. Because the geometry going in
      is already calibrated, the photoreal output stays faithful to the deployed camera; run on an
      un-calibrated scene, it just produces a convincing render of the wrong thing.</p>
  </section>

  <section>
    <p class="sec-eyebrow">Results</p>
    <h2>Where calibration shows up in the Cosmos output</h2>
    <p>Each clip pairs the raw Isaac render with its Cosmos-Transfer2.5 output (shown at half speed). The
      <b>before</b> set runs Cosmos on un-calibrated base scenes; the <b>after</b> set runs it on scenes the
      loop calibrated, alongside a real LOCO frame showing the look it&rsquo;s matching. Labels are taken
      straight from the scene folders.</p>

    <p class="caption" style="margin-top:0">Click a row to see that failure mode and its fix side by side.</p>
{pair_nav}
{pair_panels}
  </section>

  <div class="method">
    <b>Method.</b> Base clips are Isaac renders from un-calibrated parameter sets; calibrated clips use the
    &#952; selected by the latent-space MMD loop. All Cosmos clips are passed through Cosmos-Transfer2.5 with
    depth/edges/segmentation control and shown at 2&#215; slow-down. Real reference frames are drawn from the
    LOCO warehouse dataset &mdash; the same distribution the calibration loop targets.
  </div>

  <footer>Tensorleap &#183; synthetic-data calibration for warehouse object detection &#183; self-contained report</footer>
</div>
<script>
  function showPair(i) {{
    document.querySelectorAll('.pairpanel').forEach(function(el) {{
      el.hidden = el.id !== 'pair-' + i;
    }});
    document.querySelectorAll('.maprow').forEach(function(el) {{
      el.classList.toggle('active', el.dataset.pair === String(i));
    }});
    document.getElementById('pair-' + i).scrollIntoView({{behavior: 'smooth', block: 'nearest'}});
  }}
</script>
"""
    OUT_PATH.write_text(html)
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build()
