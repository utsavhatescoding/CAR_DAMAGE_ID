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

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from ultralytics import YOLO


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

.estimate-hero {
    padding: 1.15rem 1.2rem;
    background:
        linear-gradient(115deg, rgba(255,182,39,.13), transparent 62%),
        var(--panel);
    border: 1px solid rgba(255,182,39,.35);
    border-left: 4px solid var(--amber);
    margin: .55rem 0 1rem;
}

.estimate-hero .amount {
    margin-top: .25rem;
    color: var(--paper);
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(1.55rem, 4vw, 2.35rem);
    font-weight: 700;
    letter-spacing: -.025em;
}

.estimate-hero .note {
    margin-top: .35rem;
    color: var(--muted);
    font-size: .76rem;
    line-height: 1.5;
}

.cost-line {
    padding: .9rem 1rem;
    margin-bottom: .55rem;
    background: var(--panel);
    border: 1px solid var(--line);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}

.cost-line .cost-name {
    color: var(--paper);
    font-family: 'IBM Plex Mono', monospace;
    font-size: .72rem;
    font-weight: 600;
    text-transform: uppercase;
}

.cost-line .cost-detail {
    margin-top: .24rem;
    color: var(--muted);
    font-size: .68rem;
}

.cost-line .cost-value {
    color: var(--amber);
    font-family: 'IBM Plex Mono', monospace;
    font-size: .78rem;
    font-weight: 600;
    white-space: nowrap;
}

.prototype-notice {
    padding: .8rem .9rem;
    background: rgba(142,151,158,.08);
    border: 1px solid var(--line);
    color: var(--muted);
    font-size: .7rem;
    line-height: 1.55;
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
# PROTOTYPE REPAIR ESTIMATOR
# Illustrative NPR ranges only. These are deliberately separated
# from model confidence because confidence is not damage severity.
# ============================================================
VEHICLE_CATALOG = {
    "Maruti Suzuki": {
        "Alto / S-Presso": 0.85,
        "Wagon R / Celerio": 0.90,
        "Swift / Baleno": 1.00,
        "Brezza / Grand Vitara": 1.15,
    },
    "Hyundai": {
        "Grand i10 / Santro": 0.95,
        "i20": 1.05,
        "Venue": 1.15,
        "Creta": 1.25,
    },
    "Tata": {
        "Tiago / Tigor": 0.95,
        "Punch": 1.05,
        "Nexon": 1.15,
        "Safari / Harrier": 1.35,
    },
    "Mahindra": {
        "XUV300 / 3XO": 1.10,
        "Bolero": 1.10,
        "Scorpio": 1.25,
        "Thar / XUV700": 1.40,
    },
    "Kia": {
        "Picanto": 1.00,
        "Sonet": 1.15,
        "Seltos": 1.25,
        "Sportage": 1.45,
    },
    "Toyota": {
        "Yaris / Corolla": 1.20,
        "Raize": 1.20,
        "RAV4": 1.45,
        "Fortuner": 1.55,
    },
    "Honda": {
        "Brio / Amaze": 1.00,
        "City": 1.15,
        "WR-V / Elevate": 1.25,
        "CR-V": 1.50,
    },
    "Other / not listed": {
        "Compact / hatchback": 1.00,
        "Sedan": 1.10,
        "SUV / crossover": 1.25,
        "Premium / luxury": 1.65,
    },
}

DAMAGE_LABOUR_RANGES = {
    "dent": (4000, 12000),
    "scratch": (2500, 8000),
    "crack": (6000, 18000),
    "glass shatter": (3500, 8000),
    "lamp broken": (1500, 4000),
    "tire flat": (800, 2500),
}

PART_PRICE_RANGES = {
    "Bumper": (12000, 35000),
    "Door panel": (25000, 65000),
    "Fender": (15000, 38000),
    "Bonnet / hood": (30000, 80000),
    "Quarter panel": (30000, 75000),
    "Windshield / glass": (15000, 50000),
    "Headlamp / tail lamp": (8000, 40000),
    "Tyre": (10000, 32000),
    "Side mirror": (8000, 28000),
    "Other exterior panel": (15000, 50000),
}

DEFAULT_PART_BY_DAMAGE = {
    "glass shatter": "Windshield / glass",
    "lamp broken": "Headlamp / tail lamp",
    "tire flat": "Tyre",
}

INSURANCE_PACKAGES = {
    "No coverage": {
        "coverage": 0.00,
        "deductible": 0,
        "cap": 0,
        "description": "Customer pays the complete estimated repair range.",
    },
    "Essential Care": {
        "coverage": 0.50,
        "deductible": 5000,
        "cap": 30000,
        "description": "Illustrative 50% support after a NPR 5,000 deductible.",
    },
    "Smart Protect": {
        "coverage": 0.70,
        "deductible": 3000,
        "cap": 80000,
        "description": "Illustrative 70% support after a NPR 3,000 deductible.",
    },
    "Total Shield": {
        "coverage": 0.90,
        "deductible": 1500,
        "cap": 200000,
        "description": "Illustrative 90% support after a NPR 1,500 deductible.",
    },
}


def npr(value):
    return f"NPR {value:,.0f}"


def estimate_finding(damage, part, action, vehicle_multiplier):
    labour_low, labour_high = DAMAGE_LABOUR_RANGES.get(
        damage.lower(), (3500, 12000)
    )
    part_low, part_high = PART_PRICE_RANGES[part]

    if action == "Repair & refinish":
        # Part ranges provide a transparent reference, but are not charged
        # when the selected action is repair rather than replacement.
        cost_low, cost_high = labour_low, labour_high
    else:
        cost_low = labour_low + part_low
        cost_high = labour_high + part_high

    return (
        int(round(cost_low * vehicle_multiplier / 100) * 100),
        int(round(cost_high * vehicle_multiplier / 100) * 100),
        int(round(part_low * vehicle_multiplier / 100) * 100),
        int(round(part_high * vehicle_multiplier / 100) * 100),
    )


def insurance_scenario(total, package):
    if package["coverage"] <= 0:
        return 0, total
    eligible_after_deductible = max(0, total - package["deductible"])
    insurer_share = min(
        package["cap"], eligible_after_deductible * package["coverage"]
    )
    return int(round(insurer_share)), int(round(total - insurer_share))


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


def resolve_proprietary_yolo26_checkpoint():
    bundled_candidates = [
        BASE_DIR / "models" / "proprietary_yolo26_stage1_best.pt",
        BASE_DIR / "proprietary_yolo26_stage1_best.pt",
    ]
    for candidate in bundled_candidates:
        if candidate.exists() and candidate.stat().st_size >= 45 * 1024 * 1024:
            return candidate

    model_url = secret_or_environment("PROPRIETARY_YOLO26_MODEL_URL")
    if not model_url:
        raise RuntimeError(
            "Our YOLO26 model is not configured. Add "
            "PROPRIETARY_YOLO26_MODEL_URL to Streamlit Secrets using the "
            "Google Drive sharing link for the Stage 1 best.pt file."
        )

    model_dir = Path.home() / ".cache" / "cardd_vision"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "proprietary_yolo26_stage1_best.pt"
    download_checkpoint(model_path, model_url, minimum_mb=45)
    return model_path


@st.cache_resource(show_spinner=False)
def load_proprietary_yolo26m():
    return YOLO(str(resolve_proprietary_yolo26_checkpoint()))


def secret_or_environment(name: str):
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except (KeyError, FileNotFoundError):
        return None


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
        mask = np.asarray(
            Image.fromarray((mask >= 0.5).astype(np.uint8) * 255).resize(
                (width, height),
                Image.Resampling.NEAREST,
            )
        )
    return mask >= 0.5


def draw_accepted_detections(image_bgr, detections):
    canvas = Image.fromarray(image_bgr[:, :, ::-1]).convert("RGBA")
    colours = [
        (255, 182, 39),
        (120, 210, 90),
        (235, 90, 90),
        (55, 135, 215),
        (190, 90, 210),
        (225, 215, 65),
    ]

    for index, detection in enumerate(detections):
        colour = colours[index % len(colours)]
        mask = np.asarray(detection["mask"], dtype=bool)
        mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        mask_rgba[mask] = (*colour, 92)
        canvas = Image.alpha_composite(canvas, Image.fromarray(mask_rgba, "RGBA"))

        # Add a strong mask boundary without requiring OpenCV on Streamlit Cloud.
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
        boundary_width = max(3, min(11, int(round(min(mask.shape) * 0.006))))
        if boundary_width % 2 == 0:
            boundary_width += 1
        dilated = np.asarray(mask_image.filter(ImageFilter.MaxFilter(boundary_width)))
        eroded = np.asarray(mask_image.filter(ImageFilter.MinFilter(boundary_width)))
        boundary = np.logical_and(dilated > 0, eroded == 0)
        boundary_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        boundary_rgba[boundary] = (*colour, 245)
        canvas = Image.alpha_composite(
            canvas, Image.fromarray(boundary_rgba, "RGBA")
        )

    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    title_size = max(24, min(72, int(round(min(width, height) * 0.040))))
    detail_size = max(18, min(52, int(round(title_size * 0.70))))

    def load_font(size, bold=False):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
        for font_path in candidates:
            try:
                return ImageFont.truetype(font_path, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    title_font = load_font(title_size, bold=True)
    detail_font = load_font(detail_size)

    for index, detection in enumerate(detections):
        colour = colours[index % len(colours)]
        x1, y1, x2, y2 = [int(round(v)) for v in detection["box"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        line_width = max(4, min(12, int(round(min(width, height) * 0.005))))
        draw.rectangle((x1, y1, x2, y2), outline=(*colour, 255), width=line_width)

        finding = f'F-{index + 1:02d}  {detection["name"].upper()}'
        confidence = f'CONFIDENCE  {detection["confidence"]:.0%}'
        pad_x = max(10, int(round(title_size * 0.34)))
        pad_y = max(7, int(round(title_size * 0.22)))
        line_gap = max(3, int(round(title_size * 0.10)))
        title_box = draw.textbbox((0, 0), finding, font=title_font)
        detail_box = draw.textbbox((0, 0), confidence, font=detail_font)
        panel_width = max(
            title_box[2] - title_box[0], detail_box[2] - detail_box[0]
        ) + 2 * pad_x
        panel_height = (
            title_box[3] - title_box[1]
            + detail_box[3] - detail_box[1]
            + line_gap
            + 2 * pad_y
        )

        label_x = min(max(0, x1), max(0, width - panel_width))
        label_y = y1 - panel_height - line_width
        if label_y < 0:
            label_y = min(height - panel_height, y1 + line_width)
        label_y = max(0, label_y)

        panel = (
            label_x,
            label_y,
            min(width, label_x + panel_width),
            min(height, label_y + panel_height),
        )
        draw.rounded_rectangle(
            panel,
            radius=max(5, int(round(title_size * 0.18))),
            fill=(10, 14, 18, 238),
            outline=(*colour, 255),
            width=max(2, line_width // 2),
        )
        text_x = label_x + pad_x
        title_y = label_y + pad_y - title_box[1]
        draw.text(
            (text_x, title_y),
            finding,
            font=title_font,
            fill=(248, 249, 250, 255),
            stroke_width=max(1, title_size // 28),
            stroke_fill=(0, 0, 0, 220),
        )
        detail_y = (
            label_y + pad_y + title_box[3] - title_box[1] + line_gap
            - detail_box[1]
        )
        draw.text(
            (text_x, detail_y),
            confidence,
            font=detail_font,
            fill=(*colour, 255),
        )

    return np.asarray(canvas.convert("RGB"))[:, :, ::-1]


def run_scan(image, confidence):
    """Run the published Cloudwhynot damage-detection behavior.

    Both the silhouette model and damage model inspect the full image at the
    Ultralytics default 640-pixel resolution. A damage is accepted when its
    mask has any intersection with the global silhouette mask. Importantly,
    failure to identify a COCO class named exactly ``car`` never blocks the
    damage model.
    """
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

    global_vehicle_mask = np.zeros((height, width), dtype=bool)
    silhouette_count = 0
    if vehicle_result.masks is not None:
        for mask_tensor in vehicle_result.masks.data:
            global_vehicle_mask |= mask_at_size(mask_tensor, width, height)
            silhouette_count += 1

    car_count = 0
    if vehicle_result.boxes is not None:
        for box in vehicle_result.boxes:
            class_id = int(box.cls[0])
            class_name = str(vehicle_result.names[class_id]).lower().strip()
            car_count += int(class_name == "car")

    damage_result = load_cloudwhynot_yolo26m().predict(
        source=original_bgr,
        conf=confidence,
        imgsz=640,
        retina_masks=True,
        verbose=False,
    )[0]

    detections = []
    if damage_result.boxes is not None and damage_result.masks is not None:
        for index, box in enumerate(damage_result.boxes):
            damage_mask = mask_at_size(
                damage_result.masks.data[index], width, height
            )
            intersection = np.logical_and(damage_mask, global_vehicle_mask)
            intersection_area = int(intersection.sum())
            if intersection_area == 0:
                continue

            class_id = int(box.cls[0])
            global_box = [float(v) for v in box.xyxy[0].tolist()]
            damage_area = max(int(damage_mask.sum()), 1)
            detections.append(
                {
                    "name": clean_damage_name(damage_result.names[class_id]),
                    "confidence": float(box.conf[0]),
                    "box": global_box,
                    "crop": get_damage_crop(image, global_box),
                    # Cloudwhynot displays the original damage polygon after
                    # verifying that it intersects the silhouette.
                    "mask": damage_mask,
                    "vehicle_overlap": intersection_area / damage_area,
                }
            )

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    plotted = draw_accepted_detections(original_bgr, detections)
    output_image = Image.fromarray(plotted[:, :, ::-1])
    scan_time = time.perf_counter() - start_time
    pipeline_info = {
        "mode": "Cloudwhynot reference",
        "resolution": 640,
        "cars_found": car_count,
        "silhouettes_found": silhouette_count,
        "image_scope": "Full original image",
        "overlap_rule": "Any silhouette intersection",
    }
    return output_image, detections, scan_time, pipeline_info


def run_proprietary_yolo_scan(image, confidence):
    """Run our Stage 1 YOLO26m-seg checkpoint using its tested pipeline."""
    start_time = time.perf_counter()
    original_bgr = np.ascontiguousarray(np.array(image)[:, :, ::-1])
    height, width = original_bgr.shape[:2]

    result = load_proprietary_yolo26m().predict(
        source=original_bgr,
        conf=confidence,
        iou=0.70,
        imgsz=896,
        retina_masks=True,
        verbose=False,
    )[0]

    detections = []
    if result.boxes is not None:
        for index, box in enumerate(result.boxes):
            xyxy = [float(value) for value in box.xyxy[0].tolist()]
            class_id = int(box.cls[0])

            if result.masks is not None and index < len(result.masks.data):
                mask = mask_at_size(result.masks.data[index], width, height)
            else:
                mask = np.zeros((height, width), dtype=bool)
                x1, y1, x2, y2 = [int(value) for value in xyxy]
                mask[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = True

            detections.append(
                {
                    "name": clean_damage_name(result.names[class_id]),
                    "confidence": float(box.conf[0]),
                    "box": xyxy,
                    "crop": get_damage_crop(image, xyxy),
                    "mask": mask,
                }
            )

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    plotted = draw_accepted_detections(original_bgr, detections)
    output_image = Image.fromarray(plotted[:, :, ::-1])
    scan_time = time.perf_counter() - start_time
    pipeline_info = {
        "resolution": 896,
        "iou_threshold": 0.70,
        "post_filter": "None",
        "image_scope": "Full original image",
        "checkpoint": "Stage 1 best.pt",
    }
    return output_image, detections, scan_time, pipeline_info


def _box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _mask_iou(mask_a, mask_b):
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    if intersection == 0:
        return 0.0
    union = int(np.logical_or(mask_a, mask_b).sum())
    return intersection / union if union else 0.0


def _predict_proprietary_region(
    model,
    region_bgr,
    full_image,
    full_width,
    full_height,
    confidence,
    offset_x=0,
    offset_y=0,
    source="full",
):
    """Predict one region and restore all coordinates to the full image."""
    result = model.predict(
        source=region_bgr,
        conf=confidence,
        iou=0.70,
        imgsz=896,
        retina_masks=True,
        verbose=False,
    )[0]

    region_height, region_width = region_bgr.shape[:2]
    detections = []
    if result.boxes is None:
        return detections

    for index, box in enumerate(result.boxes):
        local_box = [float(value) for value in box.xyxy[0].tolist()]
        global_box = [
            local_box[0] + offset_x,
            local_box[1] + offset_y,
            local_box[2] + offset_x,
            local_box[3] + offset_y,
        ]

        if result.masks is not None and index < len(result.masks.data):
            local_mask = mask_at_size(
                result.masks.data[index], region_width, region_height
            )
        else:
            local_mask = np.zeros((region_height, region_width), dtype=bool)
            x1, y1, x2, y2 = [int(value) for value in local_box]
            local_mask[
                max(0, y1):min(region_height, y2),
                max(0, x1):min(region_width, x2),
            ] = True

        full_mask = np.zeros((full_height, full_width), dtype=bool)
        full_mask[
            offset_y:offset_y + region_height,
            offset_x:offset_x + region_width,
        ] = local_mask

        class_id = int(box.cls[0])
        detections.append(
            {
                "name": clean_damage_name(result.names[class_id]),
                "confidence": float(box.conf[0]),
                "box": global_box,
                "crop": get_damage_crop(full_image, global_box),
                "mask": full_mask,
                "sources": {source},
            }
        )

    return detections


def _primary_car_box(image_bgr, width, height):
    """Return a reliable primary-car box, or None for close-ups/failures."""
    result = load_vehicle_segmenter().predict(
        source=image_bgr,
        conf=0.25,
        imgsz=640,
        retina_masks=True,
        verbose=False,
    )[0]
    if result.boxes is None:
        return None

    candidates = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = str(result.names[class_id]).lower().strip()
        if class_name != "car":
            continue
        xyxy = [float(value) for value in box.xyxy[0].tolist()]
        x1, y1, x2, y2 = xyxy
        area_ratio = (
            max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(width * height, 1)
        )
        if area_ratio >= 0.12:
            candidates.append((area_ratio, xyxy))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _four_overlapping_tiles(box, width, height, overlap=0.18):
    """Create four bounded overlapping close-up regions inside one car box."""
    x1, y1, x2, y2 = box
    padding = 0.04 * max(x2 - x1, y2 - y1)
    x1, y1 = max(0, int(x1 - padding)), max(0, int(y1 - padding))
    x2, y2 = min(width, int(x2 + padding)), min(height, int(y2 + padding))
    region_width, region_height = x2 - x1, y2 - y1

    tile_width = min(region_width, int(round(region_width * (0.5 + overlap / 2))))
    tile_height = min(
        region_height, int(round(region_height * (0.5 + overlap / 2)))
    )
    x_starts = sorted({x1, max(x1, x2 - tile_width)})
    y_starts = sorted({y1, max(y1, y2 - tile_height)})

    tiles = []
    for tile_y in y_starts:
        for tile_x in x_starts:
            tiles.append(
                (
                    tile_x,
                    tile_y,
                    min(width, tile_x + tile_width),
                    min(height, tile_y + tile_height),
                )
            )
    return tiles


def _merge_multiscale_detections(detections):
    """Merge same-class full/tile duplicates while retaining model confidence."""
    merged = []
    for candidate in sorted(
        detections, key=lambda item: item["confidence"], reverse=True
    ):
        duplicate = None
        for existing in merged:
            if existing["name"] != candidate["name"]:
                continue
            if (
                _mask_iou(existing["mask"], candidate["mask"]) >= 0.25
                or _box_iou(existing["box"], candidate["box"]) >= 0.55
            ):
                duplicate = existing
                break

        if duplicate is None:
            merged.append(candidate)
            continue

        duplicate["mask"] = np.logical_or(
            duplicate["mask"], candidate["mask"]
        )
        duplicate["box"] = [
            min(duplicate["box"][0], candidate["box"][0]),
            min(duplicate["box"][1], candidate["box"][1]),
            max(duplicate["box"][2], candidate["box"][2]),
            max(duplicate["box"][3], candidate["box"][3]),
        ]
        duplicate["sources"].update(candidate["sources"])

    for detection in merged:
        detection["confirmed_multiscale"] = (
            "full" in detection["sources"]
            and any(source.startswith("tile") for source in detection["sources"])
        )
    return sorted(merged, key=lambda item: item["confidence"], reverse=True)


def run_multiscale_yolo_scan(image, confidence):
    """Experimental full-image + adaptive close-up inference pipeline."""
    start_time = time.perf_counter()

    # Bound memory use on CPU-hosted Streamlit while keeping useful phone detail.
    analysis_image = image.copy()
    analysis_image.thumbnail((2400, 2400), Image.Resampling.LANCZOS)
    original_bgr = np.ascontiguousarray(np.array(analysis_image)[:, :, ::-1])
    height, width = original_bgr.shape[:2]
    model = load_proprietary_yolo26m()

    raw_detections = _predict_proprietary_region(
        model,
        original_bgr,
        analysis_image,
        width,
        height,
        confidence,
        source="full",
    )

    primary_car = _primary_car_box(original_bgr, width, height)
    tiles = []
    if primary_car is not None and max(width, height) >= 1200:
        tiles = _four_overlapping_tiles(primary_car, width, height)

    tile_confidence = min(0.90, max(0.35, confidence + 0.10))
    for tile_index, (x1, y1, x2, y2) in enumerate(tiles, start=1):
        tile_bgr = original_bgr[y1:y2, x1:x2]
        if tile_bgr.size == 0:
            continue
        raw_detections.extend(
            _predict_proprietary_region(
                model,
                tile_bgr,
                analysis_image,
                width,
                height,
                tile_confidence,
                offset_x=x1,
                offset_y=y1,
                source=f"tile-{tile_index}",
            )
        )

    detections = _merge_multiscale_detections(raw_detections)
    for detection in detections:
        detection["crop"] = get_damage_crop(
            analysis_image, detection["box"]
        )

    plotted = draw_accepted_detections(original_bgr, detections)
    output_image = Image.fromarray(plotted[:, :, ::-1])
    scan_time = time.perf_counter() - start_time
    pipeline_info = {
        "resolution": 896,
        "image_scope": "Full image + adaptive vehicle tiles",
        "tile_count": len(tiles),
        "tile_threshold": tile_confidence,
        "raw_detections": len(raw_detections),
        "merged_detections": len(detections),
        "multiscale_confirmed": sum(
            detection["confirmed_multiscale"] for detection in detections
        ),
        "vehicle_localized": primary_car is not None,
        "analysis_size": f"{width} × {height}px",
    }
    return output_image, detections, scan_time, pipeline_info


def run_selected_model(model_key, image, confidence):
    if model_key == "proprietary_yolo":
        return run_proprietary_yolo_scan(image, confidence)
    if model_key == "proprietary_multiscale":
        return run_multiscale_yolo_scan(image, confidence)
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
    "Use our Stage 1 model for the exact full-image pipeline validated in testing."
)

with st.expander("Detection settings", expanded=False):
    model_choice = st.radio(
        "Inspection model",
        [
            "Our YOLO26 Stage 1",
            "Our YOLO26 Multi-Scale · Experimental",
            "Cloudwhynot YOLO26",
        ],
        horizontal=True,
        help=(
            "Stage 1 is the unchanged benchmark. Multi-Scale runs Stage 1 "
            "on the full image and up to four vehicle close-ups. Cloudwhynot "
            "uses its published silhouette-intersection approach."
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

MODEL_OPTIONS = {
    "Our YOLO26 Stage 1": (
        "proprietary_yolo",
        "YOLO26m-seg · Our Stage 1 best checkpoint",
    ),
    "Cloudwhynot YOLO26": (
        "cloud_yolo",
        "YOLO26m-seg · Cloudwhynot pipeline",
    ),
    "Our YOLO26 Multi-Scale · Experimental": (
        "proprietary_multiscale",
        "YOLO26m-seg · Full image + adaptive close-ups",
    ),
}
selected_model_key, selected_model_label = MODEL_OPTIONS[model_choice]

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
                "Running our YOLO26 Stage 1 model at 896 px..."
                if selected_model_key == "proprietary_yolo"
                else (
                    "Running full-image and adaptive close-up analysis..."
                    if selected_model_key == "proprietary_multiscale"
                    else "Running the Cloudwhynot reference pipeline..."
                )
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
            if result.get("model_key") == "proprietary_yolo":
                st.write(
                    f"**Inference resolution:** "
                    f"{pipeline_info.get('resolution', 896)}px"
                )
                st.write("**Image scope:** Full original image")
                st.write(
                    f"**IoU threshold:** "
                    f"{pipeline_info.get('iou_threshold', 0.70):.0%}"
                )
            elif result.get("model_key") == "proprietary_multiscale":
                st.write("**Pipeline:** Multi-scale experimental")
                st.write(
                    f"**Analysis size:** "
                    f"{pipeline_info.get('analysis_size', '—')}"
                )
                st.write(
                    f"**Close-up tiles:** "
                    f"{pipeline_info.get('tile_count', 0)}"
                )
                st.write(
                    f"**Tile threshold:** "
                    f"{pipeline_info.get('tile_threshold', 0.35):.0%}"
                )
                st.write(
                    f"**Confirmed at multiple scales:** "
                    f"{pipeline_info.get('multiscale_confirmed', 0)}"
                )
            else:
                st.write(
                    f"**Pipeline:** {pipeline_info.get('mode', '—')}"
                )
                st.write(
                    f"**Inference resolution:** "
                    f"{pipeline_info.get('resolution', 640)}px"
                )
                st.write("**Image scope:** Full original image")
                st.write(
                    f"**Silhouette masks:** "
                    f"{pipeline_info.get('silhouettes_found', '—')}"
                )
                st.write("**Filter:** Any silhouette intersection")

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
                        "detection_source": ", ".join(
                            sorted(d.get("sources", {"full"}))
                        ),
                        "confirmed_multiscale": d.get(
                            "confirmed_multiscale", False
                        ),
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

    if detections:
        st.write("")
        st.write("")
        html_block(
            """
            <div class="section-kicker">Step 04 · Repair estimate</div>
            <div class="section-title">Configure the vehicle and repair</div>
            """
        )
        st.caption(
            "Confirm the vehicle, affected component and intended repair action. "
            "The AI supplies the damage findings; the pricing engine uses "
            "illustrative prototype rates in Nepalese rupees."
        )

        vehicle_col, model_col = st.columns(2, gap="large")
        with vehicle_col:
            vehicle_make = st.selectbox(
                "Vehicle manufacturer",
                list(VEHICLE_CATALOG),
                key=f"estimate_make_{inspection_id}",
            )
        with model_col:
            vehicle_model = st.selectbox(
                "Vehicle model / segment",
                list(VEHICLE_CATALOG[vehicle_make]),
                key=f"estimate_model_{inspection_id}_{vehicle_make}",
            )

        vehicle_multiplier = VEHICLE_CATALOG[vehicle_make][vehicle_model]
        st.caption(
            f"Selected vehicle: {vehicle_make} {vehicle_model} · "
            f"prototype price factor {vehicle_multiplier:.2f}×"
        )
        st.write("")

        estimate_rows = []
        total_low = 0
        total_high = 0

        for i, detection in enumerate(detections, start=1):
            damage_name = detection["name"].lower()
            default_part = DEFAULT_PART_BY_DAMAGE.get(
                damage_name, "Other exterior panel"
            )
            part_options = list(PART_PRICE_RANGES)
            default_part_index = part_options.index(default_part)
            suggested_action = (
                "Replace component"
                if damage_name in DEFAULT_PART_BY_DAMAGE
                else "Repair & refinish"
            )

            with st.expander(
                f"F-{i:02d} · {detection['name'].title()} · "
                f"{detection['confidence']:.0%} confidence",
                expanded=i == 1,
            ):
                part_col, action_col = st.columns(2, gap="large")
                with part_col:
                    affected_part = st.selectbox(
                        "Affected component",
                        part_options,
                        index=default_part_index,
                        key=f"part_{inspection_id}_{i}",
                    )
                with action_col:
                    action_options = [
                        "Repair & refinish",
                        "Replace component",
                    ]
                    repair_action = st.selectbox(
                        "Repair action",
                        action_options,
                        index=action_options.index(suggested_action),
                        key=f"action_{inspection_id}_{i}",
                    )

                low, high, part_low, part_high = estimate_finding(
                    damage_name,
                    affected_part,
                    repair_action,
                    vehicle_multiplier,
                )
                st.caption(
                    f"Reference replacement part: {npr(part_low)}–"
                    f"{npr(part_high)}. It is included only when replacement "
                    "is selected."
                )

            total_low += low
            total_high += high
            estimate_rows.append(
                {
                    "finding": f"F-{i:02d}",
                    "damage": detection["name"].title(),
                    "part": affected_part,
                    "action": repair_action,
                    "low": low,
                    "high": high,
                    "part_low": part_low,
                    "part_high": part_high,
                }
            )

        html_block(
            f"""
            <div class="estimate-hero">
                <div class="section-kicker">Estimated repair range</div>
                <div class="amount">{npr(total_low)} – {npr(total_high)}</div>
                <div class="note">
                    {html.escape(vehicle_make)} · {html.escape(vehicle_model)} ·
                    {len(estimate_rows)} configured finding(s)
                </div>
            </div>
            """
        )

        for row in estimate_rows:
            html_block(
                f"""
                <div class="cost-line">
                    <div>
                        <div class="cost-name">
                            {html.escape(row['finding'])} ·
                            {html.escape(row['damage'])}
                        </div>
                        <div class="cost-detail">
                            {html.escape(row['part'])} ·
                            {html.escape(row['action'])}
                        </div>
                    </div>
                    <div class="cost-value">
                        {npr(row['low'])} – {npr(row['high'])}
                    </div>
                </div>
                """
            )

        st.write("")
        html_block(
            """
            <div class="section-kicker">Step 05 · Coverage scenario</div>
            <div class="section-title">Compare service packages</div>
            """
        )
        selected_package_name = st.selectbox(
            "Insurance service package",
            list(INSURANCE_PACKAGES),
            index=0,
            key=f"insurance_package_{inspection_id}",
        )
        selected_package = INSURANCE_PACKAGES[selected_package_name]
        st.caption(selected_package["description"])

        insurer_low, customer_low = insurance_scenario(
            total_low, selected_package
        )
        insurer_high, customer_high = insurance_scenario(
            total_high, selected_package
        )

        coverage_a, coverage_b, coverage_c = st.columns(3)
        coverage_a.metric(
            "Estimated repair",
            f"{npr(total_low)}–{npr(total_high)}",
        )
        coverage_b.metric(
            "Package contribution",
            f"{npr(insurer_low)}–{npr(insurer_high)}",
        )
        coverage_c.metric(
            "Estimated customer cost",
            f"{npr(customer_low)}–{npr(customer_high)}",
        )

        estimate_export = pd.DataFrame(estimate_rows)
        estimate_export.insert(0, "inspection_id", inspection_id)
        estimate_export["vehicle_make"] = vehicle_make
        estimate_export["vehicle_model"] = vehicle_model
        estimate_export["insurance_package"] = selected_package_name
        estimate_export["estimated_total_low_npr"] = total_low
        estimate_export["estimated_total_high_npr"] = total_high
        estimate_export["estimated_customer_low_npr"] = customer_low
        estimate_export["estimated_customer_high_npr"] = customer_high

        st.download_button(
            "Download prototype estimate CSV",
            data=estimate_export.to_csv(index=False).encode("utf-8"),
            file_name=f"{inspection_id}_prototype_estimate.csv",
            mime="text/csv",
            use_container_width=True,
        )

        html_block(
            """
            <div class="prototype-notice">
                PROTOTYPE ONLY · Rates, parts and coverage packages are dummy
                assumptions for product testing. This is not a workshop quote,
                insurer policy, claim decision or guarantee of coverage. A human
                assessor must confirm the damaged part, repair method, parts
                availability, labour and applicable policy terms.
            </div>
            """
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
