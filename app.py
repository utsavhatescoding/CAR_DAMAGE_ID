import base64
import gc
import html
import io
import os
import time
import textwrap
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


RF_CLASS_NAMES = [
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat",
]


# ============================================================
# APP / BRAND
# ============================================================
BASE_DIR = Path(__file__).parent
ASSET_DIR = BASE_DIR / "assets"
ICON_PATH = ASSET_DIR / "nvi_icon.png"

page_icon = Image.open(ICON_PATH) if ICON_PATH.exists() else "🛡️"

st.set_page_config(
    page_title="Nepal Vehicle Inspector | AI Damage Inspection",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# DESIGN SYSTEM
# Inspired by app-2.py: dark diagnostic instrument panel,
# amber warning-light accent, mono readouts, viewfinder frames.
# Core model / detection logic is unchanged.
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #0E1113;
    --panel: #171B1E;
    --panel-2: #1E2327;
    --panel-3: #242A2F;
    --paper: #F0EEE7;
    --muted: #8E979E;
    --line: rgba(240,238,231,.10);
    --line-strong: rgba(240,238,231,.17);
    --amber: #FFB627;
    --amber-dim: rgba(255,182,39,.18);
    --red: #E14B5A;
    --red-dim: rgba(225,75,90,.15);
    --teal: #34C9B8;
    --teal-dim: rgba(52,201,184,.15);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"],
.stApp {
    color: var(--paper);
    background:
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 39px,
            rgba(240,238,231,.025) 40px
        ),
        repeating-linear-gradient(
            90deg,
            transparent,
            transparent 39px,
            rgba(240,238,231,.025) 40px
        ),
        var(--bg);
}

/* Hide Streamlit chrome completely so it never overlaps the header. */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
#MainMenu,
footer {
    display: none !important;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.2rem !important;
    padding-bottom: 3.5rem !important;
}

h1, h2, h3, h4,
[data-testid="stMarkdownContainer"] strong {
    color: var(--paper);
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -0.025em;
}

h1 {
    font-size: clamp(2.35rem, 5vw, 4.35rem) !important;
    line-height: .98 !important;
    font-weight: 700 !important;
}

h2 {
    font-size: 1.65rem !important;
    font-weight: 650 !important;
}

h3 {
    font-size: 1.08rem !important;
    font-weight: 650 !important;
}

p,
.stCaption,
[data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
}

hr,
[data-testid="stDivider"] {
    border-color: var(--line) !important;
}

/* ---------- custom product header ---------- */
.nvi-brand {
    display: flex;
    align-items: center;
    gap: .85rem;
    padding-bottom: 1.25rem;
    margin-bottom: 1.7rem;
    border-bottom: 1px solid var(--line);
}

.nvi-brand img {
    width: 48px;
    height: 48px;
    object-fit: contain;
}

.nvi-brand-mark {
    width: 44px;
    height: 44px;
    border: 1px solid var(--line-strong);
    display: grid;
    place-items: center;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--amber);
    background: var(--panel);
}

.nvi-brand-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.12rem;
    line-height: 1.05;
    font-weight: 700;
    color: var(--paper);
}

.nvi-brand-sub {
    margin-top: .22rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .66rem;
    letter-spacing: .09em;
    color: var(--muted);
    text-transform: uppercase;
}

/* ---------- hero ---------- */
.hero {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: end;
    gap: 1rem;
    padding-bottom: 1.6rem;
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--line);
}

.eyebrow {
    display: flex;
    align-items: center;
    gap: .5rem;
    margin: 0 0 .7rem 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .68rem;
    letter-spacing: .14em;
    color: var(--amber);
    text-transform: uppercase;
}

.eyebrow::before {
    content: "";
    width: 7px;
    height: 7px;
    background: var(--amber);
    box-shadow: 0 0 7px var(--amber);
}

.hero-title {
    margin: 0;
    max-width: 830px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.35rem, 5vw, 4.35rem);
    line-height: .98;
    letter-spacing: -.035em;
    font-weight: 700;
    color: var(--paper);
}

.hero-title span {
    color: var(--amber);
}

.hero-copy {
    max-width: 760px;
    margin: 1rem 0 0 0;
    font-size: .98rem;
    line-height: 1.65;
    color: var(--muted);
}

.status-chip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .68rem;
    letter-spacing: .09em;
    color: var(--teal);
    border: 1px solid rgba(52,201,184,.28);
    background: rgba(52,201,184,.06);
    padding: .45rem .68rem;
    white-space: nowrap;
}

.status-chip .dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--teal);
    margin-right: .42rem;
}

/* ---------- labels / native widgets ---------- */
[data-testid="stWidgetLabel"] p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .69rem !important;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--muted) !important;
}

.section-kicker {
    margin: 0 0 .4rem 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .68rem;
    letter-spacing: .13em;
    text-transform: uppercase;
    color: var(--amber);
}

.section-title {
    margin: 0 0 .25rem 0;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 650;
    font-size: 1.45rem;
    color: var(--paper);
}

/* ---------- radio / model selector ---------- */
[data-testid="stRadio"] > div {
    gap: .65rem;
}

[data-testid="stRadio"] label {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 3px;
    padding: .78rem .92rem;
    transition: border-color .15s ease, background .15s ease;
}

[data-testid="stRadio"] label:hover {
    border-color: rgba(255,182,39,.35);
}

[data-testid="stRadio"] label:has(input:checked) {
    background: #1B2024;
    border-color: rgba(255,182,39,.60);
    box-shadow: inset 3px 0 0 var(--amber);
}

/* ---------- expander / slider ---------- */
[data-testid="stExpander"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 3px;
}

[data-testid="stSlider"] [role="slider"] {
    background: var(--amber) !important;
    border: 2px solid var(--panel) !important;
    box-shadow: 0 0 0 4px var(--amber-dim);
}

/* ---------- uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: var(--panel);
    border: 1.5px dashed rgba(255,182,39,.34);
    border-radius: 4px;
    padding: 1.15rem;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(255,182,39,.70);
    background: #1A1F23;
}

[data-testid="stFileUploaderDropzone"] button {
    background: transparent !important;
    color: var(--amber) !important;
    border: 1px solid rgba(255,182,39,.35) !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
}

/* ---------- buttons ---------- */
.stButton > button,
.stDownloadButton > button {
    min-height: 48px;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .78rem !important;
    font-weight: 600 !important;
    letter-spacing: .055em;
    text-transform: uppercase;
}

.stButton > button[kind="primary"] {
    background: var(--amber) !important;
    border: 1px solid var(--amber) !important;
    color: #181200 !important;
    box-shadow: none !important;
}

.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span {
    color: #181200 !important;
}

.stButton > button[kind="primary"]:hover {
    background: #FFC451 !important;
    border-color: #FFC451 !important;
    box-shadow: 0 0 0 3px var(--amber-dim), 0 0 18px var(--amber-dim) !important;
}

/* ---------- image viewfinder ---------- */
.frame-label {
    margin: 0 0 .5rem 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .67rem;
    letter-spacing: .13em;
    color: var(--muted);
    text-transform: uppercase;
}

.viewfinder {
    position: relative;
    padding: 10px;
    background: var(--panel);
    border: 1px solid var(--line);
    overflow: hidden;
}

.viewfinder img {
    display: block;
    width: 100%;
    height: auto;
}

.corner {
    position: absolute;
    width: 18px;
    height: 18px;
    border-color: var(--amber);
    z-index: 2;
}

.corner-tl { top: 7px; left: 7px; border-top: 2px solid; border-left: 2px solid; }
.corner-tr { top: 7px; right: 7px; border-top: 2px solid; border-right: 2px solid; }
.corner-bl { bottom: 7px; left: 7px; border-bottom: 2px solid; border-left: 2px solid; }
.corner-br { bottom: 7px; right: 7px; border-bottom: 2px solid; border-right: 2px solid; }

.scanline {
    position: absolute;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--amber), transparent);
    opacity: .70;
    z-index: 2;
    animation: sweep 2.5s linear infinite;
}

@keyframes sweep {
    from { top: 0%; }
    to { top: 100%; }
}

@media (prefers-reduced-motion: reduce) {
    .scanline { animation: none; }
}

/* ---------- loaded image meta ---------- */
.meta-panel {
    height: 100%;
    background: var(--panel);
    border: 1px solid var(--line);
    padding: 1rem;
}

.meta-row {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    padding: .58rem 0;
    border-bottom: 1px solid var(--line);
    font-family: 'IBM Plex Mono', monospace;
    font-size: .72rem;
}

.meta-row:last-child {
    border-bottom: 0;
}

.meta-key {
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .07em;
}

.meta-value {
    color: var(--paper);
    text-align: right;
}

/* ---------- metrics ---------- */
[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 2px;
    padding: .9rem 1rem;
    box-shadow: none;
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .67rem !important;
    letter-spacing: .08em;
    text-transform: uppercase;
}

[data-testid="stMetricValue"] {
    color: var(--paper) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 650 !important;
}

/* ---------- tabs ---------- */
[data-baseweb="tab-list"] {
    gap: .85rem;
    border-bottom: 1px solid var(--line);
}

[data-baseweb="tab"] {
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: .72rem;
    letter-spacing: .06em;
    text-transform: uppercase;
}

[aria-selected="true"][data-baseweb="tab"] {
    color: var(--amber) !important;
}

/* ---------- findings ticket ---------- */
.ticket {
    display: flex;
    align-items: center;
    gap: .85rem;
    background: var(--panel);
    border: 1px solid var(--line);
    border-left: 3px solid var(--red);
    padding: .72rem .9rem;
    margin-bottom: .55rem;
}

.ticket .finding-id {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .65rem;
    color: var(--muted);
    min-width: 42px;
}

.ticket .name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .78rem;
    letter-spacing: .035em;
    text-transform: uppercase;
    min-width: 155px;
    color: var(--paper);
}

.ticket .bar-track {
    flex: 1;
    height: 6px;
    background: var(--panel-3);
    min-width: 60px;
    overflow: hidden;
}

.ticket .bar-fill {
    height: 100%;
    background: var(--red);
}

.ticket .score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .74rem;
    color: var(--muted);
    min-width: 48px;
    text-align: right;
}

.severity {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .61rem;
    letter-spacing: .07em;
    padding: .15rem .4rem;
    min-width: 42px;
    text-align: center;
}

.sev-high { background: var(--red-dim); color: var(--red); }
.sev-med { background: var(--amber-dim); color: var(--amber); }
.sev-low { background: rgba(142,151,158,.14); color: var(--muted); }

.clear-banner {
    padding: .85rem 1rem;
    background: var(--teal-dim);
    border: 1px solid rgba(52,201,184,.32);
    color: var(--teal);
    font-family: 'IBM Plex Mono', monospace;
    font-size: .75rem;
    letter-spacing: .045em;
}


/* ---------- native image frame ---------- */
[data-testid="stImage"] {
    background: var(--panel);
    border: 1px solid var(--line-strong);
    padding: 9px;
}

[data-testid="stImage"] img {
    display: block;
    width: 100%;
    border-radius: 0 !important;
}

/* ---------- dark results table ---------- */
.results-table-wrap {
    width: 100%;
    overflow-x: auto;
    border: 1px solid var(--line-strong);
    background: var(--panel);
}

.results-table {
    width: 100%;
    border-collapse: collapse;
    min-width: 680px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .72rem;
}

.results-table th {
    text-align: left;
    padding: .78rem .85rem;
    color: var(--amber);
    background: var(--panel-2);
    border-bottom: 1px solid var(--line-strong);
    text-transform: uppercase;
    letter-spacing: .07em;
    font-weight: 600;
}

.results-table td {
    padding: .78rem .85rem;
    color: var(--paper);
    border-bottom: 1px solid var(--line);
}

.results-table tbody tr:last-child td {
    border-bottom: 0;
}

.results-table tbody tr:hover {
    background: rgba(255,182,39,.035);
}

/* ---------- dataframe / alerts / progress ---------- */
[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 2px;
    overflow: hidden;
}

[data-testid="stAlert"] {
    border-radius: 2px;
}

[data-testid="stProgress"] > div > div > div {
    background: var(--amber) !important;
}

/* ---------- footer ---------- */
.footer-strip {
    margin-top: 2.3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--line);
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .65rem;
    color: var(--muted);
    letter-spacing: .06em;
    text-transform: uppercase;
}

/* ---------- mobile ---------- */
@media (max-width: 768px) {
    .block-container {
        max-width: 100% !important;
        padding: 1.1rem .85rem 2.5rem !important;
    }

    .nvi-brand {
        margin-bottom: 1.2rem;
        padding-bottom: 1rem;
    }

    .nvi-brand img,
    .nvi-brand-mark {
        width: 40px;
        height: 40px;
    }

    .nvi-brand-name {
        font-size: 1rem;
    }

    .hero {
        grid-template-columns: 1fr;
        align-items: start;
        gap: .85rem;
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.2rem;
        line-height: 1.00;
    }

    .hero-copy {
        font-size: .91rem;
        line-height: 1.55;
    }

    .status-chip {
        width: fit-content;
    }

    /* Streamlit columns must stack on phones */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: .8rem !important;
    }

    [data-testid="column"] {
        width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }

    /* Keep the model selector fully visible on mobile */
    [data-testid="stRadio"] > div {
        flex-direction: column !important;
        gap: .48rem !important;
    }

    [data-testid="stRadio"] label {
        width: 100% !important;
        min-height: 48px;
        padding: .68rem .75rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        padding: .9rem;
    }

    [data-testid="stFileUploaderDropzone"] button,
    .stButton > button,
    .stDownloadButton > button {
        width: 100% !important;
        min-height: 46px;
    }

    .viewfinder {
        padding: 7px;
    }

    .ticket {
        flex-wrap: wrap;
        gap: .55rem .7rem;
    }

    .ticket .name {
        min-width: calc(100% - 110px);
        flex: 1;
    }

    .ticket .bar-track {
        order: 5;
        width: 100%;
        flex-basis: 100%;
    }

    .ticket .score {
        margin-left: auto;
    }

    [data-baseweb="tab-list"] {
        overflow-x: auto;
        white-space: nowrap;
        gap: 1rem;
    }

    .meta-panel {
        padding: .85rem;
    }

    .meta-row {
        font-size: .68rem;
    }
}


@media (max-width: 768px) {
    [data-testid="stImage"] {
        padding: 6px;
    }

    .results-table {
        font-size: .66rem;
        min-width: 620px;
    }

    .results-table th,
    .results-table td {
        padding: .65rem .7rem;
    }
}

@media (max-width: 420px) {
    .block-container {
        padding-left: .7rem !important;
        padding-right: .7rem !important;
    }

    .hero-title {
        font-size: 1.95rem;
    }

    .nvi-brand-sub {
        font-size: .59rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PRESENTATION HELPERS
# ============================================================
def html_block(content: str):
    st.markdown(textwrap.dedent(content).strip(), unsafe_allow_html=True)


def image_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def render_viewfinder(img: Image.Image, label: str, scanning: bool = False):
    # Native Streamlit image rendering is deliberate here.
    # It avoids Markdown/base64 being shown as visible HTML on Streamlit Cloud.
    html_block(f'<div class="frame-label">{label}</div>')
    st.image(img, use_container_width=True)


def severity_for(score: float):
    if score >= 0.75:
        return "HIGH", "sev-high"
    if score >= 0.50:
        return "MED", "sev-med"
    return "LOW", "sev-low"


def render_ticket(index: int, detection: dict):
    score = detection["confidence"]
    html_block(
        f"""
        <div class="ticket">
            <span class="finding-id">F-{index:02d}</span>
            <span class="name">{detection["name"]}</span>
            <div class="bar-track">
                <div class="bar-fill" style="width:{score*100:.0f}%"></div>
            </div>
            <span class="score">{score:.1%}</span>
        </div>
        """
    )


def render_summary_table(rows):
    if not rows:
        return

    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['Finding']))}</td>"
            f"<td>{html.escape(str(row['Damage']))}</td>"
            f"<td>{html.escape(str(row['Confidence']))}</td>"
            f"<td>{html.escape(str(row['Region']))}</td>"
            "</tr>"
        )

    html_block(
        """
        <div class="results-table-wrap">
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Finding</th>
                        <th>Damage</th>
                        <th>Confidence</th>
                        <th>Region</th>
                    </tr>
                </thead>
                <tbody>
        """
        + "".join(body)
        + """
                </tbody>
            </table>
        </div>
        """
    )


# ============================================================
# MODELS
# ============================================================
def download_checkpoint(model_path: Path, url: str, minimum_mb: int = 40):
    minimum_bytes = minimum_mb * 1024 * 1024
    if model_path.exists() and model_path.stat().st_size >= minimum_bytes:
        return

    model_path.unlink(missing_ok=True)
    temporary_path = model_path.with_suffix(".part")
    temporary_path.unlink(missing_ok=True)

    if "drive.google.com" in url:
        import gdown

        downloaded = gdown.download(
            url=url,
            output=str(temporary_path),
            quiet=True,
            fuzzy=True,
        )
        if downloaded is None:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Google Drive could not download the checkpoint. Confirm that "
                "General access is set to 'Anyone with the link'."
            )
    else:
        urllib.request.urlretrieve(url, str(temporary_path))

    if temporary_path.stat().st_size < minimum_bytes:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("A model checkpoint download was incomplete.")

    temporary_path.replace(model_path)


@st.cache_resource(show_spinner=False)
def load_vehicle_segmenter():
    model_dir = Path.home() / ".cache" / "cardd_vision"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "yolo26m_seg_vehicle.pt"
    download_checkpoint(
        model_path,
        "https://github.com/cloudwhynot/"
        "car-damage-detection-yolo/raw/refs/heads/main/"
        "models/yolo26m-seg.pt",
    )
    return YOLO(str(model_path))


@st.cache_resource(show_spinner=False)
def load_cloudwhynot_yolo26m():
    model_dir = Path.home() / ".cache" / "cardd_vision"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "cloudwhynot_yolo26m_seg_best.pt"

    download_checkpoint(
        model_path,
        "https://github.com/cloudwhynot/"
        "car-damage-detection-yolo/raw/refs/heads/main/"
        "models/damage_model/weights/best.pt",
    )

    return YOLO(str(model_path))


def secret_or_environment(name: str):
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except (KeyError, FileNotFoundError):
        return None


def resolve_rf_checkpoint():
    bundled_candidates = [
        BASE_DIR / "models" / "checkpoint_best_ema.pth",
        BASE_DIR / "checkpoint_best_ema.pth",
    ]
    for candidate in bundled_candidates:
        if candidate.exists() and candidate.stat().st_size >= 120 * 1024 * 1024:
            return candidate

    model_url = secret_or_environment("RFDETR_MODEL_URL")
    if not model_url:
        raise RuntimeError(
            "RF-DETR is not configured. Add RFDETR_MODEL_URL to Streamlit "
            "Secrets using a permanent direct-download link to "
            "checkpoint_best_ema.pth."
        )

    model_dir = Path.home() / ".cache" / "cardd_vision"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "checkpoint_best_ema.pth"
    download_checkpoint(model_path, model_url, minimum_mb=120)
    return model_path


@st.cache_resource(show_spinner=False)
def load_our_rfdetr():
    import torch
    from rfdetr import RFDETR

    checkpoint = resolve_rf_checkpoint()
    runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    return RFDETR.from_checkpoint(
        str(checkpoint),
        trust_checkpoint=True,
        device=runtime_device,
    )


def clean_damage_name(name):
    replacements = {
        "glass_shatter": "Shattered Glass",
        "shattered_glass": "Shattered Glass",
        "lamp_broken": "Broken Lamp",
        "broken_lamp": "Broken Lamp",
        "tire_flat": "Flat Tire",
        "flat_tire": "Flat Tire",
    }
    name = str(name).lower().strip()
    if name in replacements:
        return replacements[name]
    return name.replace("_", " ").title()


def get_damage_crop(image, xyxy, padding=35):
    x1, y1, x2, y2 = map(int, xyxy)
    width, height = image.size
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    return image.crop((x1, y1, x2, y2))


def mask_at_size(mask, width, height):
    mask = mask.detach().cpu().numpy()
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask >= 0.5


def select_target_car(vehicle_result, width, height):
    if vehicle_result.boxes is None or vehicle_result.masks is None:
        raise ValueError(
            "No car was found. Use one clear photo with the target car fully visible."
        )

    candidates = []
    for index, box in enumerate(vehicle_result.boxes):
        class_id = int(box.cls[0])
        class_name = str(vehicle_result.names[class_id]).lower().strip()
        if class_name != "car":
            continue

        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        mask = mask_at_size(vehicle_result.masks.data[index], width, height)
        candidates.append((area, [x1, y1, x2, y2], mask))

    if not candidates:
        raise ValueError("No car was found. This version currently inspects cars only.")

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2], len(candidates)


def padded_crop_box(box, width, height, padding_ratio=0.08):
    x1, y1, x2, y2 = box
    padding = max(x2 - x1, y2 - y1) * padding_ratio
    return (
        max(0, int(x1 - padding)),
        max(0, int(y1 - padding)),
        min(width, int(x2 + padding)),
        min(height, int(y2 + padding)),
    )


def draw_accepted_detections(image_bgr, detections):
    canvas = image_bgr.copy()
    overlay = image_bgr.copy()
    colours = [
        (39, 182, 255),
        (90, 210, 120),
        (90, 90, 235),
        (215, 135, 55),
        (210, 90, 190),
        (65, 215, 225),
    ]

    for index, detection in enumerate(detections):
        overlay[detection["mask"]] = colours[index % len(colours)]

    canvas = cv2.addWeighted(overlay, 0.42, canvas, 0.58, 0)

    for index, detection in enumerate(detections):
        colour = colours[index % len(colours)]
        contours, _ = cv2.findContours(
            detection["mask"].astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(canvas, contours, -1, colour, 2)

        x1, y1, x2, y2 = [int(v) for v in detection["box"]]
        label = f'{detection["name"]} {detection["confidence"]:.0%}'
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            canvas,
            label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            colour,
            2,
            cv2.LINE_AA,
        )

    return canvas


def run_scan(image, confidence, overlap_threshold=0.60):
    start_time = time.perf_counter()
    original_bgr = np.ascontiguousarray(np.array(image)[:, :, ::-1])
    height, width = original_bgr.shape[:2]

    vehicle_result = load_vehicle_segmenter().predict(
        source=original_bgr,
        conf=0.25,
        imgsz=640,
        retina_masks=True,
        verbose=False,
    )[0]
    vehicle_box, vehicle_mask, car_count = select_target_car(
        vehicle_result, width, height
    )

    crop_x1, crop_y1, crop_x2, crop_y2 = padded_crop_box(
        vehicle_box, width, height
    )
    crop_bgr = original_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_vehicle_mask = vehicle_mask[crop_y1:crop_y2, crop_x1:crop_x2]

    # Keep damage located on lamps, tyres, mirrors and outer panel boundaries.
    dilation_size = max(5, int(round(max(crop_bgr.shape[:2]) * 0.012)))
    if dilation_size % 2 == 0:
        dilation_size += 1
    kernel = np.ones((dilation_size, dilation_size), dtype=np.uint8)
    crop_vehicle_mask = cv2.dilate(
        crop_vehicle_mask.astype(np.uint8), kernel, iterations=1
    ).astype(bool)

    damage_result = load_cloudwhynot_yolo26m().predict(
        source=crop_bgr,
        conf=confidence,
        imgsz=896,
        retina_masks=True,
        verbose=False,
    )[0]

    detections = []
    if damage_result.boxes is not None and damage_result.masks is not None:
        crop_height, crop_width = crop_bgr.shape[:2]
        for index, box in enumerate(damage_result.boxes):
            damage_mask = mask_at_size(
                damage_result.masks.data[index], crop_width, crop_height
            )
            damage_area = int(damage_mask.sum())
            if damage_area == 0:
                continue

            overlap_ratio = float(
                np.logical_and(damage_mask, crop_vehicle_mask).sum() / damage_area
            )
            if overlap_ratio < overlap_threshold:
                continue

            filtered_crop_mask = np.logical_and(damage_mask, crop_vehicle_mask)
            full_mask = np.zeros((height, width), dtype=bool)
            full_mask[crop_y1:crop_y2, crop_x1:crop_x2] = filtered_crop_mask

            class_id = int(box.cls[0])
            local_box = [float(v) for v in box.xyxy[0].tolist()]
            global_box = [
                local_box[0] + crop_x1,
                local_box[1] + crop_y1,
                local_box[2] + crop_x1,
                local_box[3] + crop_y1,
            ]
            detections.append(
                {
                    "name": clean_damage_name(damage_result.names[class_id]),
                    "confidence": float(box.conf[0]),
                    "box": global_box,
                    "crop": get_damage_crop(image, global_box),
                    "mask": full_mask,
                    "vehicle_overlap": overlap_ratio,
                }
            )

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    plotted = draw_accepted_detections(original_bgr, detections)
    output_image = Image.fromarray(plotted[:, :, ::-1])
    scan_time = time.perf_counter() - start_time
    pipeline_info = {
        "cars_found": car_count,
        "target_box": vehicle_box,
        "crop_box": [crop_x1, crop_y1, crop_x2, crop_y2],
        "overlap_threshold": overlap_threshold,
    }
    return output_image, detections, scan_time, pipeline_info


def run_rf_scan(image, confidence):
    """Run our RF-DETR checkpoint directly on the full, original image."""
    start_time = time.perf_counter()
    original_bgr = np.ascontiguousarray(np.array(image)[:, :, ::-1])
    height, width = original_bgr.shape[:2]

    predictions = load_our_rfdetr().predict(
        image,
        threshold=confidence,
        shape=(624, 624),
        include_source_image=False,
    )

    class_ids = (
        np.asarray(predictions.class_id, dtype=int)
        if predictions.class_id is not None
        else np.array([], dtype=int)
    )
    scores = (
        np.asarray(predictions.confidence, dtype=float)
        if predictions.confidence is not None
        else np.array([], dtype=float)
    )
    boxes = np.asarray(predictions.xyxy, dtype=float)
    masks = predictions.mask
    checkpoint_names = predictions.data.get("class_name")

    detections = []
    for index, (class_id, score, box) in enumerate(
        zip(class_ids, scores, boxes)
    ):
        if checkpoint_names is not None and index < len(checkpoint_names):
            raw_name = str(checkpoint_names[index]).strip()
        elif 0 <= class_id < len(RF_CLASS_NAMES):
            raw_name = RF_CLASS_NAMES[class_id]
        else:
            raw_name = f"class {class_id}"

        if masks is None or index >= len(masks):
            mask = np.zeros((height, width), dtype=bool)
            x1, y1, x2, y2 = [int(v) for v in box]
            mask[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = True
        else:
            mask = np.asarray(masks[index])
            if mask.shape != (height, width):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = mask.astype(bool)

        detections.append(
            {
                "name": clean_damage_name(raw_name),
                "confidence": float(score),
                "box": [float(v) for v in box],
                "crop": get_damage_crop(image, box),
                "mask": mask,
            }
        )

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    plotted = draw_accepted_detections(original_bgr, detections)
    output_image = Image.fromarray(plotted[:, :, ::-1])
    scan_time = time.perf_counter() - start_time
    pipeline_info = {
        "resolution": 624,
        "post_filter": "None",
        "image_scope": "Full original image",
    }
    return output_image, detections, scan_time, pipeline_info


def run_selected_model(model_key, image, confidence):
    if model_key == "rf_detr":
        return run_rf_scan(image, confidence)
    if model_key == "cloud_yolo":
        return run_scan(image, confidence)
    raise ValueError(f"Unknown model: {model_key}")


# ============================================================
# STATE
# ============================================================
if "inspection_result" not in st.session_state:
    st.session_state.inspection_result = None
if "inspection_source_name" not in st.session_state:
    st.session_state.inspection_source_name = None
if "inspection_id" not in st.session_state:
    st.session_state.inspection_id = None


# ============================================================
# BRAND HEADER
# ============================================================
if ICON_PATH.exists():
    icon_uri = image_to_data_uri(Image.open(ICON_PATH).convert("RGBA"))
    brand_icon_html = f'<img src="{icon_uri}" alt="Nepal Vehicle Inspector">'
else:
    brand_icon_html = '<div class="nvi-brand-mark">NVI</div>'

html_block(
    f"""
    <div class="nvi-brand">
        {brand_icon_html}
        <div>
            <div class="nvi-brand-name">Nepal Vehicle Inspector</div>
            <div class="nvi-brand-sub">AI-powered vehicle damage assessment</div>
        </div>
    </div>
    """
)


# ============================================================
# HERO
# ============================================================
html_block(
    """
    <div class="hero">
        <div>
            <div class="eyebrow">Vehicle damage assessment</div>
            <div class="hero-title">See the damage.<br><span>Know what matters.</span></div>
            <p class="hero-copy">
                Upload a vehicle photo for a fast visual assessment with annotated damage,
                confidence scores and cropped evidence.
            </p>
        </div>
        <div class="status-chip"><span class="dot"></span>INSPECTION SYSTEM READY</div>
    </div>
    """
)


# ============================================================
# ASSESSMENT
# ============================================================
html_block(
    """
    <div class="section-kicker">Step 01 · Configure</div>
    <div class="section-title">Inspect a vehicle</div>
    """
)
st.caption(
    "Set the detection threshold and provide one clear exterior photo. "
    "The system automatically isolates the main car before damage analysis."
)

with st.expander("Detection settings", expanded=False):
    model_choice = st.radio(
        "Inspection model",
        ["Our RF-DETR", "Cloudwhynot YOLO26"],
        horizontal=True,
        help=(
            "RF-DETR runs directly on the full image. Cloudwhynot first "
            "isolates the main car, then runs YOLO26 damage segmentation."
        ),
    )
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Lower thresholds show more possible damage but can increase false positives.",
    )
    st.caption(
        f"Current threshold: {confidence:.0%}. "
        "Confidence is model certainty, not physical severity."
    )

selected_model_key = (
    "rf_detr" if model_choice == "Our RF-DETR" else "cloud_yolo"
)
selected_model_label = (
    "RF-DETR Seg Medium · Our trained checkpoint"
    if selected_model_key == "rf_detr"
    else "YOLO26m-seg · Cloudwhynot pipeline"
)

st.write("")
html_block('<div class="section-kicker">Step 02 · Load image</div>')

input_method = st.radio(
    "Image source",
    ["Upload photo", "Use camera"],
    horizontal=True,
)

if input_method == "Upload photo":
    image_file = st.file_uploader(
        "Upload vehicle photo",
        type=["jpg", "jpeg", "png", "webp"],
        help="JPG, PNG or WEBP. Clear daylight images work best.",
        label_visibility="collapsed",
    )
else:
    image_file = st.camera_input(
        "Take a clear photo of one car",
        help="Keep the target car large, sharp and fully visible.",
    )

if image_file is None:
    html_block(
        """
        <div style="
            margin-top:.35rem;
            padding:1.45rem;
            text-align:center;
            background:var(--panel);
            border:1px dashed var(--line-strong);
            color:var(--muted);
            font-family:'IBM Plex Mono',monospace;
            font-size:.72rem;
            letter-spacing:.04em;">
            AWAITING INPUT · Upload a clear exterior vehicle photo to begin
        </div>
        """
    )
else:
    image = Image.open(image_file).convert("RGB")
    st.write("")

    preview, details = st.columns([1.5, 1], gap="large", vertical_alignment="center")

    with preview:
        render_viewfinder(image, "Input feed", scanning=True)

    with details:
        html_block(
            f"""
            <div class="meta-panel">
                <div class="section-kicker">Ready to analyze</div>
                <div class="meta-row">
                    <span class="meta-key">Model</span>
                    <span class="meta-value">{selected_model_label}</span>
                </div>
                <div class="meta-row">
                    <span class="meta-key">Threshold</span>
                    <span class="meta-value">{confidence:.0%}</span>
                </div>
                <div class="meta-row">
                    <span class="meta-key">Image</span>
                    <span class="meta-value">{image.width} × {image.height}px</span>
                </div>
                <div class="meta-row">
                    <span class="meta-key">Output</span>
                    <span class="meta-value">Annotated + evidence</span>
                </div>
            </div>
            """
        )
        st.write("")
        run = st.button(
            "Run inspection",
            type="primary",
            use_container_width=True,
        )

    if run:
        try:
            spinner_message = (
                "Running RF-DETR on the full image..."
                if selected_model_key == "rf_detr"
                else "Isolating the vehicle and running YOLO26..."
            )
            with st.spinner(spinner_message):
                output_image, detections, scan_time, pipeline_info = (
                    run_selected_model(selected_model_key, image, confidence)
                )

            st.session_state.inspection_result = {
                "original": image.copy(),
                "annotated": output_image,
                "detections": detections,
                "scan_time": scan_time,
                "model_key": selected_model_key,
                "model_name": selected_model_label,
                "threshold": confidence,
                "pipeline_info": pipeline_info,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state.inspection_source_name = getattr(
                image_file, "name", "camera_capture.jpg"
            )
            st.session_state.inspection_id = datetime.now().strftime(
                "NVI-%y%m%d-%H%M%S"
            )
        except Exception as exc:
            st.error(f"Inspection failed: {exc}")


# ============================================================
# RESULTS
# ============================================================
result = st.session_state.inspection_result

if result is not None:
    detections = result["detections"]
    highest = max((d["confidence"] for d in detections), default=0.0)
    unique_types = len({d["name"] for d in detections})
    inspection_id = st.session_state.inspection_id or "NVI-SESSION"

    st.write("")
    st.write("")
    html_block(
        f"""
        <div class="section-kicker">Step 03 · Inspection report</div>
        <div class="section-title">
            {"Damage findings detected" if detections else "No visible damage detected"}
        </div>
        """
    )
    st.caption(
        f"{inspection_id} · {result['model_name']} · "
        f"{result['threshold']:.0%} threshold · {result['scan_time']:.2f}s"
    )

    st.write("")
    m1, m2, m3 = st.columns(3)
    m1.metric("Findings", len(detections))
    m2.metric("Damage types", unique_types)
    m3.metric("Highest confidence", f"{highest:.0%}")

    st.write("")
    overview_tab, evidence_tab, detail_tab = st.tabs(
        ["Overview", "Evidence", "Details"]
    )

    with overview_tab:
        st.write("")
        before, after = st.columns(2, gap="large")

        with before:
            render_viewfinder(result["original"], "Original")

        with after:
            render_viewfinder(result["annotated"], "AI assessment")

        st.write("")
        html_block('<div class="section-kicker">Findings</div>')

        if detections:
            for i, detection in enumerate(detections, start=1):
                render_ticket(i, detection)

            st.write("")
            summary_rows = []
            for i, d in enumerate(detections, start=1):
                x1, y1, x2, y2 = d["box"]
                summary_rows.append(
                    {
                        "Finding": f"F-{i:02d}",
                        "Damage": d["name"],
                        "Confidence": f"{d['confidence']:.1%}",
                        "Region": f"({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})",
                    }
                )
            render_summary_table(summary_rows)
        else:
            html_block(
                '<div class="clear-banner">✓ NO VISIBLE DAMAGE DETECTED ABOVE THE SELECTED THRESHOLD</div>'
            )

    with evidence_tab:
        st.write("")
        if not detections:
            st.info("No evidence regions to review.")
        else:
            st.caption(
                "Each item shows the exact detected region and the model confidence."
            )
            st.write("")

            for i, d in enumerate(detections, start=1):
                with st.container(border=True):
                    crop, info = st.columns(
                        [1, 1.25],
                        gap="large",
                        vertical_alignment="center",
                    )

                    with crop:
                        render_viewfinder(d["crop"], f"Evidence F-{i:02d}")

                    with info:
                        label, sev_class = severity_for(d["confidence"])
                        html_block(
                            f"""
                            <div class="section-kicker">Finding F-{i:02d}</div>
                            <div class="section-title">{d["name"]}</div>
                            """
                        )
                        st.metric("Confidence", f"{d['confidence']:.1%}")
                        st.progress(
                            min(max(d["confidence"], 0.0), 1.0)
                        )
                        x1, y1, x2, y2 = d["box"]
                        st.caption(
                            f"Region: ({x1:.0f}, {y1:.0f}) → "
                            f"({x2:.0f}, {y2:.0f})"
                        )
                        st.caption(
                            "Confidence reflects model certainty; "
                            "it is not a repair-cost or severity score."
                        )

    with detail_tab:
        st.write("")
        left, right = st.columns(2, gap="large")

        with left:
            html_block('<div class="section-kicker">Assessment</div>')
            st.write(f"**Inspection ID:** {inspection_id}")
            st.write(f"**Timestamp:** {result['timestamp']}")
            st.write(
                f"**Source file:** "
                f"{st.session_state.inspection_source_name}"
            )
            st.write(
                f"**Image size:** "
                f"{result['original'].width} × "
                f"{result['original'].height}px"
            )

        with right:
            html_block('<div class="section-kicker">Model</div>')
            st.write(f"**Model:** {result['model_name']}")
            st.write(f"**Threshold:** {result['threshold']:.0%}")
            st.write(
                f"**Inference time:** {result['scan_time']:.3f}s"
            )
            st.write(f"**Detected regions:** {len(detections)}")
            pipeline_info = result.get("pipeline_info", {})
            if result.get("model_key") == "rf_detr":
                st.write(
                    f"**Inference resolution:** "
                    f"{pipeline_info.get('resolution', 624)}px"
                )
                st.write("**Image scope:** Full original image")
                st.write("**Additional filtering:** None")
            else:
                st.write(
                    f"**Cars found:** {pipeline_info.get('cars_found', '—')}"
                )
                st.write(
                    "**Vehicle-overlap filter:** "
                    f"{pipeline_info.get('overlap_threshold', 0.60):.0%}"
                )

        if detections:
            export_rows = []
            for i, d in enumerate(detections, start=1):
                x1, y1, x2, y2 = d["box"]
                export_rows.append(
                    {
                        "inspection_id": inspection_id,
                        "finding_id": f"F-{i:02d}",
                        "damage": d["name"],
                        "confidence": round(d["confidence"], 4),
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "x2": round(x2, 1),
                        "y2": round(y2, 1),
                    }
                )

            csv_bytes = (
                pd.DataFrame(export_rows)
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                "Download findings CSV",
                data=csv_bytes,
                file_name=f"{inspection_id}_findings.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.write("")
    st.caption(
        "AI-assisted visual screening only. A qualified human inspector "
        "should review findings before safety, valuation, repair or "
        "insurance decisions."
    )


# ============================================================
# FOOTER
# ============================================================
html_block(
    """
    <div class="footer-strip">
        <span>Nepal Vehicle Inspector · AI damage assessment</span>
        <span>NVI · 2026</span>
    </div>
    """
)
