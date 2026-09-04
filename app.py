from io import BytesIO
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

import streamlit as st
import pandas as pd
import plotly.express as px
import html
import re

st.set_page_config(
    page_title="TradePulse Nepal | Nepal Trade Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Design / CSS
# -----------------------------

# A restrained but distinctive financial-intelligence palette. Keeping the main
# content light preserves readability for large tables while the hero/navigation
# use deeper colors for a premium terminal-like identity.
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    "#355CFF", "#16B8C6", "#EF476F", "#7C5CFC", "#10B981", "#F59E0B"
]

st.markdown("""
<style>
    :root {
        --tp-bg: #F5F7FB;
        --tp-card: rgba(255, 255, 255, 0.92);
        --tp-card-solid: #FFFFFF;
        --tp-ink: #0A1020;
        --tp-muted: #667085;
        --tp-line: rgba(20, 34, 66, 0.10);
        --tp-navy: #0A1020;
        --tp-indigo: #355CFF;
        --tp-cyan: #16B8C6;
        --tp-violet: #7C5CFC;
        --tp-green: #0F9F76;
        --tp-amber: #D98B12;
        --tp-red: #D93F66;
        --tp-shadow: 0 18px 55px rgba(25, 39, 74, 0.10);
    }

    html { scroll-behavior: smooth; }

    .stApp {
        background:
            radial-gradient(circle at 8% 2%, rgba(53, 92, 255, 0.10), transparent 26%),
            radial-gradient(circle at 93% 7%, rgba(22, 184, 198, 0.09), transparent 24%),
            linear-gradient(180deg, #FBFCFF 0%, var(--tp-bg) 48%, #F7F8FC 100%);
        color: var(--tp-ink);
    }

    [data-testid="stHeader"] {
        background: rgba(251, 252, 255, 0.72);
        backdrop-filter: blur(14px);
    }

    .block-container {
        padding-top: 1.15rem;
        padding-bottom: 3.5rem;
        max-width: 1420px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background:
            radial-gradient(circle at 0% 0%, rgba(53, 92, 255, 0.10), transparent 30%),
            #FFFFFF;
        border-right: 1px solid var(--tp-line);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.1rem;
    }

    .side-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px 13px;
        margin: 0 0 15px 0;
        border: 1px solid var(--tp-line);
        border-radius: 18px;
        background: rgba(248, 250, 255, 0.92);
        box-shadow: 0 10px 28px rgba(20, 34, 66, 0.06);
    }

    .side-logo, .brand-mark {
        display: grid;
        place-items: center;
        width: 42px;
        height: 42px;
        border-radius: 13px;
        color: #FFFFFF;
        font-weight: 900;
        letter-spacing: -1px;
        background: linear-gradient(135deg, var(--tp-indigo), var(--tp-cyan));
        box-shadow: 0 10px 24px rgba(53, 92, 255, 0.25);
        flex: 0 0 auto;
    }

    .side-brand-title {
        color: var(--tp-ink);
        font-size: 15px;
        line-height: 1.05;
        font-weight: 850;
        letter-spacing: -0.3px;
    }

    .side-brand-sub {
        color: var(--tp-muted);
        font-size: 11px;
        margin-top: 4px;
        letter-spacing: 0.25px;
    }

    /* Hero */
    .hero {
        position: relative;
        overflow: hidden;
        isolation: isolate;
        margin-bottom: 22px;
        padding: 34px;
        border-radius: 30px;
        color: #FFFFFF;
        background:
            radial-gradient(circle at 88% 10%, rgba(22, 184, 198, 0.38), transparent 30%),
            radial-gradient(circle at 58% 110%, rgba(124, 92, 252, 0.42), transparent 38%),
            linear-gradient(132deg, #091020 0%, #132455 49%, #153B61 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 30px 75px rgba(9, 16, 32, 0.24);
    }

    .hero::before {
        content: "";
        position: absolute;
        inset: 0;
        z-index: -1;
        opacity: 0.28;
        background-image:
            linear-gradient(rgba(255,255,255,0.08) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.08) 1px, transparent 1px);
        background-size: 44px 44px;
        mask-image: linear-gradient(to right, rgba(0,0,0,0.75), transparent 82%);
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 320px;
        height: 320px;
        right: -145px;
        top: -160px;
        border: 1px solid rgba(255,255,255,0.22);
        border-radius: 50%;
        box-shadow:
            0 0 0 36px rgba(255,255,255,0.035),
            0 0 0 76px rgba(255,255,255,0.02);
    }

    .hero-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.65fr) minmax(280px, 0.75fr);
        gap: 28px;
        align-items: stretch;
    }

    .brand-lockup {
        display: flex;
        gap: 11px;
        align-items: center;
        margin-bottom: 24px;
    }

    .brand-mark {
        width: 44px;
        height: 44px;
        border-radius: 14px;
        background: linear-gradient(135deg, #5A75FF, #24D1CD);
    }

    .brand-name {
        font-size: 14px;
        font-weight: 850;
        letter-spacing: 0.2px;
    }

    .brand-line {
        margin-top: 2px;
        font-size: 10px;
        color: rgba(255,255,255,0.66);
        text-transform: uppercase;
        letter-spacing: 1.4px;
    }

    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 13px;
        color: #8DE9E5;
        font-size: 11px;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 1.25px;
    }

    .hero-kicker::before {
        content: "";
        width: 22px;
        height: 2px;
        border-radius: 99px;
        background: #8DE9E5;
    }

    .hero-title {
        max-width: 860px;
        margin: 0 0 14px 0;
        font-size: clamp(38px, 4.6vw, 67px);
        line-height: 0.98;
        letter-spacing: -2.7px;
        font-weight: 900;
    }

    .hero-title span {
        color: #A6F3EE;
    }

    .hero-subtitle {
        max-width: 790px;
        margin: 0;
        color: rgba(255,255,255,0.78);
        font-size: 15.5px;
        line-height: 1.66;
    }

    .hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 22px;
    }

    .meta-chip, .pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 8px 11px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.14);
        color: rgba(255,255,255,0.86);
        font-size: 11px;
        font-weight: 750;
        letter-spacing: 0.25px;
    }

    .meta-chip.live::before, .pill::before {
        content: "";
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #58E1A5;
        box-shadow: 0 0 0 5px rgba(88,225,165,0.12);
    }

    .pulse-panel {
        align-self: stretch;
        padding: 19px;
        border-radius: 22px;
        background: rgba(5, 12, 30, 0.48);
        border: 1px solid rgba(255,255,255,0.13);
        backdrop-filter: blur(16px);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
    }

    .pulse-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: rgba(255,255,255,0.54);
        font-size: 10px;
        font-weight: 850;
        letter-spacing: 1.3px;
        text-transform: uppercase;
    }

    .pulse-period {
        margin: 10px 0 15px 0;
        font-size: 19px;
        line-height: 1.22;
        font-weight: 850;
        letter-spacing: -0.5px;
    }

    .pulse-row {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 2px 14px;
        padding: 11px 0;
        border-top: 1px solid rgba(255,255,255,0.10);
    }

    .pulse-row span {
        color: rgba(255,255,255,0.62);
        font-size: 11px;
        font-weight: 700;
    }

    .pulse-row strong {
        font-size: 14px;
        font-weight: 850;
    }

    .pulse-row em {
        grid-column: 1 / -1;
        color: #A6F3EE;
        font-size: 10px;
        font-style: normal;
        font-weight: 750;
    }

    /* KPI row */
    .kpi-card {
        position: relative;
        overflow: hidden;
        min-height: 138px;
        padding: 20px 21px 18px;
        border-radius: 21px;
        background: var(--tp-card);
        border: 1px solid var(--tp-line);
        box-shadow: var(--tp-shadow);
        backdrop-filter: blur(14px);
        transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }

    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: rgba(53, 92, 255, 0.20);
        box-shadow: 0 22px 60px rgba(25, 39, 74, 0.14);
    }

    .kpi-card::after {
        content: "";
        position: absolute;
        width: 78px;
        height: 78px;
        right: -28px;
        bottom: -33px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(53,92,255,0.13), rgba(22,184,198,0.02) 68%);
    }

    .kpi-label {
        margin-bottom: 10px;
        color: #778197;
        font-size: 10.5px;
        font-weight: 900;
        letter-spacing: 1.15px;
        text-transform: uppercase;
    }

    .kpi-value {
        margin-bottom: 7px;
        color: var(--tp-ink);
        font-size: clamp(25px, 2.2vw, 34px);
        line-height: 1.05;
        font-weight: 900;
        letter-spacing: -1.2px;
    }

    .kpi-note {
        color: var(--tp-muted);
        font-size: 11.5px;
        line-height: 1.45;
    }

    /* Data status strip */
    .status-shell {
        margin: 21px 0 8px 0;
        padding: 15px;
        border-radius: 21px;
        background: rgba(255,255,255,0.68);
        border: 1px solid var(--tp-line);
        box-shadow: 0 12px 36px rgba(25,39,74,0.055);
        backdrop-filter: blur(12px);
    }

    .status-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin: 0 2px 11px 2px;
    }

    .status-title {
        color: var(--tp-ink);
        font-size: 12px;
        font-weight: 900;
        letter-spacing: 0.9px;
        text-transform: uppercase;
    }

    .status-trust {
        color: var(--tp-green);
        font-size: 10px;
        font-weight: 800;
    }

    .status-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0,1fr));
        gap: 10px;
    }

    .status-card {
        min-width: 0;
        padding: 12px 13px;
        border-radius: 14px;
        background: #FFFFFF;
        border: 1px solid rgba(20,34,66,0.08);
    }

    .status-label {
        margin-bottom: 4px;
        color: #8A94A8;
        font-size: 9px;
        font-weight: 850;
        letter-spacing: .7px;
        text-transform: uppercase;
    }

    .status-value {
        overflow: hidden;
        color: #172033;
        font-size: 12px;
        font-weight: 800;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .status-note {
        margin: 10px 3px 1px 3px;
        color: #7A8497;
        font-size: 10.5px;
        line-height: 1.45;
    }

    /* Content cards */
    .section-card,
    .analyst-card,
    .executive-brief,
    .feedback-card {
        background: var(--tp-card);
        border: 1px solid var(--tp-line);
        border-radius: 21px;
        padding: 20px;
        box-shadow: 0 14px 40px rgba(25,39,74,0.07);
        backdrop-filter: blur(12px);
        line-height: 1.62;
    }

    .section-card { margin-bottom: 14px; }
    .feedback-card { margin: 15px 0; }

    .insight-card,
    .risk-card,
    .opportunity-card {
        border-radius: 18px;
        padding: 17px 18px;
        margin-bottom: 13px;
        color: var(--tp-ink);
        line-height: 1.58;
        box-shadow: 0 10px 30px rgba(25,39,74,0.045);
    }

    .insight-card {
        background: linear-gradient(135deg, #F2F6FF, #F6FBFF);
        border: 1px solid #D8E3FF;
        border-left: 4px solid var(--tp-indigo);
    }

    .risk-card {
        background: linear-gradient(135deg, #FFF3F6, #FFF8FA);
        border: 1px solid #FFD9E3;
        border-left: 4px solid var(--tp-red);
    }

    .opportunity-card {
        background: linear-gradient(135deg, #EEFCF8, #F5FFFC);
        border: 1px solid #CDEFE5;
        border-left: 4px solid var(--tp-green);
    }

    .analyst-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 13px;
        margin: 12px 0 18px 0;
    }

    .analyst-card h4,
    .feedback-card h3,
    .section-card h3 {
        margin: 0 0 8px 0;
        color: var(--tp-ink);
        letter-spacing: -0.35px;
    }

    .analyst-card p {
        margin: 0;
        color: #465166;
        font-size: 13px;
        line-height: 1.58;
    }

    .insight-badge {
        display: inline-flex;
        align-items: center;
        padding: 5px 9px;
        margin-bottom: 9px;
        border-radius: 999px;
        color: #2447D8;
        background: #EAF0FF;
        border: 1px solid #D8E3FF;
        font-size: 9.5px;
        font-weight: 900;
        letter-spacing: .8px;
        text-transform: uppercase;
    }

    .contact-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin-top: 13px;
    }

    .contact-item {
        padding: 13px 14px;
        border-radius: 14px;
        background: #F7F9FD;
        border: 1px solid var(--tp-line);
        font-size: 12px;
    }

    .contact-item b {
        display: block;
        margin-bottom: 3px;
        color: var(--tp-ink);
    }

    .tp-footer {
        margin-top: 30px;
        padding: 20px 22px;
        border: 1px solid var(--tp-line);
        border-radius: 20px;
        background: rgba(255,255,255,0.78);
        color: var(--tp-muted);
        font-size: 11.5px;
        line-height: 1.55;
        box-shadow: 0 12px 32px rgba(25,39,74,0.05);
    }

    .tp-footer b { color: var(--tp-ink); }

    .launch-state {
        position: relative;
        overflow: hidden;
        padding: 25px;
        border-radius: 22px;
        border: 1px solid rgba(53,92,255,0.16);
        background:
            radial-gradient(circle at 90% 10%, rgba(22,184,198,0.12), transparent 30%),
            linear-gradient(135deg, #F8FAFF, #F3F7FF);
        box-shadow: 0 14px 40px rgba(25,39,74,0.07);
    }

    .launch-state .launch-tag {
        display: inline-flex;
        padding: 6px 9px;
        margin-bottom: 10px;
        border-radius: 999px;
        color: #2447D8;
        background: #E8EFFF;
        font-size: 9px;
        font-weight: 900;
        letter-spacing: .9px;
        text-transform: uppercase;
    }

    .launch-state h3 {
        margin: 0 0 8px 0;
        color: var(--tp-ink);
        font-size: 22px;
        letter-spacing: -.6px;
    }

    .launch-state p {
        max-width: 820px;
        margin: 0;
        color: #596579;
        line-height: 1.6;
        font-size: 13px;
    }

    /* Streamlit-native components */
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 6px;
        overflow-x: auto;
        padding: 8px 4px 9px 4px;
        scrollbar-width: thin;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        height: 38px;
        padding: 0 14px;
        border-radius: 999px;
        color: #596579;
        background: rgba(255,255,255,0.74);
        border: 1px solid var(--tp-line);
        font-weight: 750;
        white-space: nowrap;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        color: #FFFFFF;
        background: linear-gradient(135deg, #213A8F, #355CFF);
        border-color: transparent;
        box-shadow: 0 8px 22px rgba(53,92,255,0.20);
    }

    .stDownloadButton button,
    .stButton button {
        min-height: 40px;
        border-radius: 12px !important;
        font-weight: 780 !important;
        border: 1px solid var(--tp-line) !important;
        box-shadow: 0 5px 14px rgba(25,39,74,0.05);
        transition: transform .16s ease, box-shadow .16s ease !important;
    }

    .stDownloadButton button:hover,
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 9px 22px rgba(25,39,74,0.10);
    }

    div[data-baseweb="select"] > div,
    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input {
        border-radius: 12px !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--tp-line);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 28px rgba(25,39,74,0.045);
    }

    hr {
        border: none !important;
        border-top: 1px solid var(--tp-line) !important;
    }

    h1, h2, h3 {
        color: var(--tp-ink);
        letter-spacing: -0.55px;
    }

    @media (max-width: 960px) {
        .hero-grid { grid-template-columns: 1fr; }
        .pulse-panel { max-width: 100%; }
        .status-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: .85rem;
            padding-right: .85rem;
        }

        .hero {
            padding: 23px 20px;
            border-radius: 23px;
        }

        .hero-title {
            font-size: 39px;
            letter-spacing: -1.8px;
        }

        .hero-subtitle { font-size: 14px; }
        .hero-meta { gap: 6px; }
        .meta-chip { font-size: 10px; }

        .kpi-card {
            min-height: auto;
            padding: 16px;
            margin-bottom: 8px;
        }

        .analyst-grid,
        .contact-grid,
        .status-grid { grid-template-columns: 1fr; }

        .status-value { white-space: normal; }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar + fiscal-year data discovery
# -----------------------------

MONTH_NAME_ORDER = {
    "shrawan": 1, "shravan": 1, "srawan": 1,
    "bhadra": 2, "bhadraw": 2,
    "ashwin": 3, "asoj": 3, "ashoj": 3,
    "kartik": 4, "karthik": 4,
    "mangsir": 5, "margshir": 5,
    "poush": 6, "paush": 6,
    "magh": 7,
    "falgun": 8, "phalgun": 8, "fagun": 8,
    "chaitra": 9, "chait": 9,
    "baisakh": 10, "baishakh": 10, "vaisakh": 10,
    "jestha": 11, "jeth": 11,
    "ashadh": 12, "asad": 12, "asar": 12,
}


def fiscal_year_key(folder_name):
    """Return a sortable fiscal-year key for names such as 2083_84 or 2083-84."""
    name = str(folder_name).strip()
    match = re.fullmatch(r"(\d{4})\s*[_\-/]\s*(\d{2,4})", name)
    if not match:
        return None
    start_year = int(match.group(1))
    end_text = match.group(2)
    end_year = int(end_text) if len(end_text) == 4 else int(str(start_year)[:2] + end_text)
    return start_year, end_year


def fiscal_year_display(folder_name):
    key = fiscal_year_key(folder_name)
    if not key:
        return None
    start_year, end_year = key
    return f"FY {start_year}/{str(end_year)[-2:]}"


def monthly_file_sort_key(path):
    """Sort fiscal-month workbooks reliably even when a numeric prefix is absent."""
    stem = Path(path).stem.lower().replace("-", " ").replace("_", " ")
    prefix = re.match(r"\s*(\d{1,2})\b", stem)
    if prefix:
        number = int(prefix.group(1))
        if 1 <= number <= 12:
            return number, stem

    for month_name, order in MONTH_NAME_ORDER.items():
        if month_name in stem:
            return order, stem

    return 999, stem


def clean_period_label(file_path):
    try:
        stem = Path(file_path).stem
    except Exception:
        stem = str(file_path)

    stem = stem.replace("_", " ").replace("-", " ").strip()
    stem = re.sub(r"^\s*\d{1,2}\s+", "", stem)
    stem = re.sub(r"\s+", " ", stem)
    return stem or "Not available"


def discover_active_monthly_data(root_path):
    """Select the newest fiscal-year folder, falling back to root-level Excel files."""
    root_path = Path(root_path)
    if not root_path.exists():
        return root_path, None, []

    fy_candidates = []
    for child in root_path.iterdir():
        if child.is_dir():
            key = fiscal_year_key(child.name)
            if key:
                fy_candidates.append((key, child))

    if fy_candidates:
        _, active_path = max(fy_candidates, key=lambda item: item[0])
    else:
        active_path = root_path

    files = sorted(active_path.glob("*.xlsx"), key=monthly_file_sort_key)
    active_fy = fiscal_year_display(active_path.name) if active_path != root_path else None
    return active_path, active_fy, files


st.sidebar.markdown(
    """
    <div class="side-brand">
        <div class="side-logo">TP</div>
        <div>
            <div class="side-brand-title">TradePulse Nepal</div>
            <div class="side-brand-sub">Trade intelligence · Nepal</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Data Controls")
st.sidebar.caption(
    "Upload a Department of Customs workbook for temporary analysis, or let TradePulse automatically use the latest file from the newest fiscal-year folder."
)

uploaded_file = st.sidebar.file_uploader(
    "Upload customs Excel file",
    type=["xlsx"],
    key="customs_excel_uploader"
)

app_root = Path(__file__).parent
default_file_path = app_root / "customs.xlsx"
monthly_root_path = app_root / "monthly_data"
developer_photo_path = app_root / "utsav.png"

# Discover all fiscal-year folders and let the user choose the reporting period.
fy_candidates = []
if monthly_root_path.exists():
    for child in monthly_root_path.iterdir():
        if child.is_dir():
            key = fiscal_year_key(child.name)
            if key:
                fy_candidates.append((key, child))
fy_candidates.sort(key=lambda item: item[0], reverse=True)

if fy_candidates:
    fy_labels = [fiscal_year_display(path.name) or path.name for _, path in fy_candidates]
    fy_lookup = {label: path for label, (_, path) in zip(fy_labels, fy_candidates)}

    st.markdown("### Viewing Period")
    st.caption("Choose the fiscal year and Customs release used across the dashboard.")
    period_col1, period_col2 = st.columns(2)
    with period_col1:
        selected_fy_label = st.selectbox(
            "Fiscal year", fy_labels, index=0, key="tp_fiscal_year"
        )

    monthly_data_path = fy_lookup[selected_fy_label]
    active_fy_label = selected_fy_label
    monthly_files_for_default = sorted(
        monthly_data_path.glob("*.xlsx"), key=monthly_file_sort_key
    )

    with period_col2:
        if monthly_files_for_default:
            release_labels = [clean_period_label(path) for path in monthly_files_for_default]
            release_lookup = dict(zip(release_labels, monthly_files_for_default))
            selected_release_label = st.selectbox(
                "Release",
                release_labels,
                index=len(release_labels) - 1,
                key=f"tp_release_{monthly_data_path.name}",
            )
            selected_monthly_file = release_lookup[selected_release_label]
        else:
            st.selectbox("Release", ["No releases available"], disabled=True)
            selected_release_label = "No releases available"
            selected_monthly_file = None
else:
    monthly_data_path = monthly_root_path
    active_fy_label = None
    monthly_files_for_default = sorted(
        monthly_root_path.glob("*.xlsx"), key=monthly_file_sort_key
    ) if monthly_root_path.exists() else []
    selected_monthly_file = monthly_files_for_default[-1] if monthly_files_for_default else None
    selected_release_label = clean_period_label(selected_monthly_file) if selected_monthly_file else "Not available"

latest_monthly_file = monthly_files_for_default[-1] if monthly_files_for_default else None

# Trend calculations must never look beyond the release the user selected.
# Example: selecting Kartik should use Shrawan→Kartik only, not future months from the same FY.
selected_trend_end_name = selected_monthly_file.name if selected_monthly_file is not None else None

def get_trend_files_for_period(monthly_data_path, end_file_name=None):
    path = Path(monthly_data_path)
    if not path.exists():
        return []

    files = sorted(path.glob("*.xlsx"), key=monthly_file_sort_key)
    if end_file_name:
        names = [f.name for f in files]
        if end_file_name in names:
            files = files[: names.index(end_file_name) + 1]
    return files

# Production priority:
# user upload -> user-selected fiscal-year release -> customs.xlsx fallback.
if uploaded_file is not None:
    file_source = uploaded_file
elif selected_monthly_file is not None:
    file_source = selected_monthly_file
elif default_file_path.exists():
    file_source = default_file_path
else:
    st.warning(
        "No Customs workbook was found. Add customs.xlsx, or create monthly_data/<FY>/ and place at least one .xlsx file inside it."
    )
    st.stop()


def data_source_name(file_obj):
    if hasattr(file_obj, "name"):
        return file_obj.name
    try:
        return Path(file_obj).name
    except Exception:
        return "Uploaded workbook"


def data_source_type(uploaded_file, default_file_path, selected_monthly_file, file_source):
    if uploaded_file is not None:
        return "Temporary uploaded workbook"
    if selected_monthly_file is not None and Path(file_source) == selected_monthly_file:
        return "Selected fiscal-year release"
    if default_file_path.exists() and Path(file_source) == default_file_path:
        return "Fallback customs.xlsx"
    return "Workbook"


monthly_files_count = len(monthly_files_for_default)
latest_month_label = selected_release_label
source_file_label = data_source_name(file_source)
source_type_label = data_source_type(uploaded_file, default_file_path, selected_monthly_file, file_source)

st.sidebar.markdown("---")
st.sidebar.markdown("### Data Status")
st.sidebar.write(f"**Source:** {source_type_label}")
if active_fy_label:
    st.sidebar.write(f"**Active fiscal year:** {active_fy_label}")
st.sidebar.write(f"**Selected release:** {latest_month_label}")
st.sidebar.write(f"**Monthly releases:** {monthly_files_count}")
st.sidebar.caption(f"File: {source_file_label}")
st.sidebar.info("Source values are in Rs. thousands. TradePulse displays monetary values in Rs. billion.")
st.sidebar.markdown("**Build:** TradePulse 1.0 · FY-aware")

st.sidebar.markdown("---")
st.sidebar.markdown("### Connect")
st.sidebar.markdown("[GitHub · utsavhatescoding](https://github.com/utsavhatescoding)")
st.sidebar.markdown("Email: **utsavkphuyal@gmail.com**")

# -----------------------------
# Load Excel
# -----------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def load_data_cached(source_payload, is_uploaded_file=False, file_mtime_ns=None):
    """Read the main Customs workbook with caching.

    Streamlit Cloud is slower than local. Caching prevents the full Excel workbook
    from being re-read on every small UI interaction. file_mtime_ns is only a cache
    invalidation key for path-based files.
    """
    excel_source = BytesIO(source_payload) if is_uploaded_file else source_payload
    excel = pd.ExcelFile(excel_source)

    trade = pd.read_excel(excel, sheet_name="1_Trade_Direction", header=2)
    trade.columns = trade.columns.astype(str).str.strip()

    # Some customs files use "Trade Indicators" while others use "Trade.Indicators".
    # Standardize it so the rest of the dashboard works.
    for col in trade.columns:
        clean_col = str(col).lower().replace(".", " ").replace("_", " ").strip()
        if "trade" in clean_col and "indicator" in clean_col:
            trade = trade.rename(columns={col: "Trade.Indicators"})
            break

    countries = pd.read_excel(excel, sheet_name="3_Trade_Balance_Country", header=2)
    import_partner = pd.read_excel(excel, sheet_name="4_Imports_By_Commodity_Partner", header=2)
    imports = pd.read_excel(excel, sheet_name="5_Imports_By_Commodity", header=2)
    export_partner = pd.read_excel(excel, sheet_name="6_Exports_By_Commodity_Partner", header=2)
    exports = pd.read_excel(excel, sheet_name="7_Exports_By_Commodity", header=2)
    customs = pd.read_excel(excel, sheet_name="9_Customswise_Trade", header=2)

    return trade, countries, import_partner, imports, export_partner, exports, customs


def load_data(file):
    if hasattr(file, "getvalue"):
        return load_data_cached(file.getvalue(), True)

    file_path = Path(file)
    # Include modified time in the cache key so monthly GitHub updates refresh properly.
    return load_data_cached(str(file_path.resolve()), False, file_path.stat().st_mtime_ns)


try:
    trade, countries, import_partner, imports, export_partner, exports, customs = load_data(file_source)
except Exception as e:
    st.error("Could not read the Excel file. Make sure it is the Department of Customs workbook and the sheet names are unchanged.")
    st.exception(e)
    st.stop()

# -----------------------------
# Helper functions
# -----------------------------

def rs_thousand_to_billion(value):
    return value / 1_000_000


def safe_number(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def get_trade_value(pattern, value_col):
    mask = trade["Trade.Indicators"].astype(str).str.contains(
        pattern,
        case=False,
        na=False,
        regex=True
    )

    if mask.sum() == 0:
        return 0.0

    return safe_number(trade.loc[mask, value_col].iloc[0])




def get_trade_value_from_df(trade_df, pattern, value_col):
    indicator_series = (
        trade_df["Trade.Indicators"]
        .astype(str)
        .str.lower()
        .str.replace(".", " ", regex=False)
        .str.replace("_", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    mask = indicator_series.str.contains(
        pattern,
        case=False,
        na=False,
        regex=True
    )

    if mask.sum() == 0:
        return 0.0

    return safe_number(trade_df.loc[mask, value_col].iloc[0])


@st.cache_data(show_spinner=False, ttl=3600)
def read_trade_direction_smart(file):
    # Read without assumptions so monthly files with slightly different labels still work.
    raw = pd.read_excel(file, sheet_name="1_Trade_Direction", header=None)

    header_row = None

    for i in range(len(raw)):
        row_text = " ".join(
            raw.iloc[i]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        ).lower()

        normalized = row_text.replace(".", " ").replace("_", " ")

        if "trade" in normalized and "indicator" in normalized:
            header_row = i
            break

    if header_row is None:
        raise ValueError("Could not detect header row in 1_Trade_Direction sheet.")

    trade_df = pd.read_excel(file, sheet_name="1_Trade_Direction", header=header_row)
    trade_df.columns = trade_df.columns.astype(str).str.strip()

    # Rename Trade Indicators column to one standard name
    for col in trade_df.columns:
        clean_col = str(col).lower().replace(".", " ").replace("_", " ").strip()
        if "trade" in clean_col and "indicator" in clean_col:
            trade_df = trade_df.rename(columns={col: "Trade.Indicators"})
            break

    if "Trade.Indicators" not in trade_df.columns:
        raise ValueError("Trade indicator column found, but could not standardize it.")

    return trade_df


@st.cache_data(show_spinner=False, ttl=3600)
def extract_trade_snapshot_from_file(file, file_name):
    trade_df = read_trade_direction_smart(file)

    # Find latest/current value column.
    # We ignore SN and Change columns, then use the last numeric column.
    possible_cols = []

    for col in trade_df.columns:
        col_text = str(col).strip().lower()

        if col == "Trade.Indicators":
            continue
        if "change" in col_text:
            continue
        if col_text in ["sn", "s n", "s.n", "s.no", "s no"]:
            continue

        numeric_series = pd.to_numeric(trade_df[col], errors="coerce")
        if numeric_series.notna().sum() >= 3:
            possible_cols.append(col)

    if len(possible_cols) == 0:
        raise ValueError("No usable numeric value column found.")

    selected_col = possible_cols[-1]

    imports_raw = get_trade_value_from_df(trade_df, r"^imports\b", selected_col)
    exports_raw = get_trade_value_from_df(trade_df, r"^exports\b", selected_col)
    deficit_raw = get_trade_value_from_df(trade_df, r"^trade\s+deficit\b", selected_col)
    total_trade_raw = get_trade_value_from_df(trade_df, r"^total\s+foreign\s+trade\b", selected_col)

    period_name = (
        file_name
        .replace(".xlsx", "")
        .replace(".xls", "")
        .replace("_", " ")
        .replace("-", " ")
    )

    return {
        "File": file_name,
        "Period": period_name,
        "Source Column": selected_col,
        "Imports_Billion": rs_thousand_to_billion(imports_raw),
        "Exports_Billion": rs_thousand_to_billion(exports_raw),
        "Trade_Deficit_Billion": rs_thousand_to_billion(deficit_raw),
        "Total_Trade_Billion": rs_thousand_to_billion(total_trade_raw),
        "Import_Export_Ratio": imports_raw / exports_raw if exports_raw else 0
    }



@st.cache_data(show_spinner=False, ttl=3600)
def read_product_table(file, sheet_name, value_col):
    """Read commodity import/export table from a customs workbook."""
    df = pd.read_excel(file, sheet_name=sheet_name, header=2)
    df.columns = df.columns.astype(str).str.strip()

    required_cols = ["HSCode", "Description", value_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{col} not found in {sheet_name}")

    df = df.copy()

    df["HSCode"] = (
        df["HSCode"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
        .str.zfill(8)
    )

    df["Description"] = df["Description"].astype(str).str.strip()

    df = df[
        ~df["Description"]
        .str.lower()
        .isin(["total", "grand total", "nan"])
    ]

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)
    df["Value_Billion"] = df[value_col] / 1_000_000

    df = (
        df.groupby(["HSCode", "Description"], as_index=False)["Value_Billion"]
        .sum()
    )

    return df


@st.cache_data(show_spinner=False, ttl=3600)
def build_product_movement_table(trend_files, sheet_name, value_col):
    """Build latest monthly product movement from cumulative monthly customs files."""
    monthly_tables = []
    periods = []

    for file_path in sorted(trend_files, key=monthly_file_sort_key):
        period = file_path.stem.replace("_", " ")
        periods.append(period)

        temp = read_product_table(file_path, sheet_name, value_col)
        temp["Period"] = period
        monthly_tables.append(temp)

    if len(monthly_tables) < 2:
        raise ValueError("At least two files are required for product movement analysis.")

    all_products = pd.concat(monthly_tables, ignore_index=True)

    cumulative_wide = all_products.pivot_table(
        index=["HSCode", "Description"],
        columns="Period",
        values="Value_Billion",
        aggfunc="sum",
        fill_value=0
    )

    cumulative_wide = cumulative_wide[periods]

    # Customs files are cumulative, so monthly values are differences between cumulative files.
    monthly_wide = cumulative_wide.diff(axis=1)
    monthly_wide[periods[0]] = cumulative_wide[periods[0]]

    latest_period = periods[-1]
    previous_period = periods[-2]

    result = cumulative_wide.reset_index()[["HSCode", "Description"]].copy()
    result["Latest_Cumulative_Billion"] = cumulative_wide[latest_period].values
    result["Previous_Month_Billion"] = monthly_wide[previous_period].values
    result["Latest_Month_Billion"] = monthly_wide[latest_period].values
    result["Change_Billion"] = result["Latest_Month_Billion"] - result["Previous_Month_Billion"]

    result["Growth_Percent"] = result.apply(
        lambda row: (row["Change_Billion"] / row["Previous_Month_Billion"] * 100)
        if row["Previous_Month_Billion"] != 0 else None,
        axis=1
    )

    result["Latest_Period"] = latest_period
    result["Previous_Period"] = previous_period

    return result


@st.cache_data(show_spinner=False, ttl=3600)
def build_trend_summary_for_report(monthly_data_path, end_file_name=None):
    """Create a compact trend summary for the app and PDF report from static monthly_data files."""
    try:
        monthly_data_path = Path(monthly_data_path)
        if not monthly_data_path.exists():
            return None

        trend_files = get_trend_files_for_period(monthly_data_path, end_file_name)
        if len(trend_files) < 2:
            return None

        trend_rows = []
        for file_path in trend_files:
            try:
                trend_rows.append(extract_trade_snapshot_from_file(file_path, file_path.name))
            except Exception:
                continue

        if len(trend_rows) < 2:
            return None

        trend_df = pd.DataFrame(trend_rows)
        trend_df["Order"] = range(1, len(trend_df) + 1)

        trend_df["Monthly_Imports_Billion"] = trend_df["Imports_Billion"].diff()
        trend_df["Monthly_Exports_Billion"] = trend_df["Exports_Billion"].diff()
        trend_df["Monthly_Deficit_Billion"] = trend_df["Trade_Deficit_Billion"].diff()
        trend_df["Monthly_Total_Trade_Billion"] = trend_df["Total_Trade_Billion"].diff()

        trend_df.loc[trend_df.index[0], "Monthly_Imports_Billion"] = trend_df.loc[trend_df.index[0], "Imports_Billion"]
        trend_df.loc[trend_df.index[0], "Monthly_Exports_Billion"] = trend_df.loc[trend_df.index[0], "Exports_Billion"]
        trend_df.loc[trend_df.index[0], "Monthly_Deficit_Billion"] = trend_df.loc[trend_df.index[0], "Trade_Deficit_Billion"]
        trend_df.loc[trend_df.index[0], "Monthly_Total_Trade_Billion"] = trend_df.loc[trend_df.index[0], "Total_Trade_Billion"]

        latest = trend_df.iloc[-1]
        previous = trend_df.iloc[-2]

        import_change = latest["Imports_Billion"] - previous["Imports_Billion"]
        export_change = latest["Exports_Billion"] - previous["Exports_Billion"]
        deficit_change = latest["Trade_Deficit_Billion"] - previous["Trade_Deficit_Billion"]

        if import_change > export_change:
            movement_note = "Imports increased more than exports in the latest period, which may widen trade-deficit pressure."
        elif export_change > import_change:
            movement_note = "Exports increased more than imports in the latest period, which may slightly ease trade-deficit pressure."
        else:
            movement_note = "Imports and exports moved by a similar amount in the latest period."

        summary = {
            "files_used": len(trend_df),
            "latest_period": latest["Period"],
            "previous_period": previous["Period"],
            "latest_imports": latest["Imports_Billion"],
            "latest_exports": latest["Exports_Billion"],
            "latest_deficit": latest["Trade_Deficit_Billion"],
            "latest_total_trade": latest["Total_Trade_Billion"],
            "import_change": import_change,
            "export_change": export_change,
            "deficit_change": deficit_change,
            "movement_note": movement_note,
            "top_rising_import": None,
            "top_rising_import_change": None,
            "top_rising_export": None,
            "top_rising_export_change": None,
            "top_falling_import": None,
            "top_falling_import_change": None,
            "top_falling_export": None,
            "top_falling_export_change": None,
        }

        try:
            import_product_movement = build_product_movement_table(
                trend_files=trend_files,
                sheet_name="5_Imports_By_Commodity",
                value_col="Imports_Value"
            )
            export_product_movement = build_product_movement_table(
                trend_files=trend_files,
                sheet_name="7_Exports_By_Commodity",
                value_col="Exports_Value"
            )

            import_filtered = import_product_movement[import_product_movement["Latest_Month_Billion"] >= 0.1].copy()
            export_filtered = export_product_movement[export_product_movement["Latest_Month_Billion"] >= 0.1].copy()

            if len(import_filtered) > 0:
                top_rising_import = import_filtered.sort_values("Change_Billion", ascending=False).iloc[0]
                top_falling_import = import_filtered.sort_values("Change_Billion", ascending=True).iloc[0]
                summary["top_rising_import"] = top_rising_import["Description"]
                summary["top_rising_import_change"] = top_rising_import["Change_Billion"]
                summary["top_falling_import"] = top_falling_import["Description"]
                summary["top_falling_import_change"] = top_falling_import["Change_Billion"]

            if len(export_filtered) > 0:
                top_rising_export = export_filtered.sort_values("Change_Billion", ascending=False).iloc[0]
                top_falling_export = export_filtered.sort_values("Change_Billion", ascending=True).iloc[0]
                summary["top_rising_export"] = top_rising_export["Description"]
                summary["top_rising_export_change"] = top_rising_export["Change_Billion"]
                summary["top_falling_export"] = top_falling_export["Description"]
                summary["top_falling_export_change"] = top_falling_export["Change_Billion"]
        except Exception:
            pass

        return summary

    except Exception:
        return None

def short_text(text, max_len=58):
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def kpi_card(label, value, note):
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    safe_note = html.escape(str(note))
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{safe_label}</div>
            <div class="kpi-value">{safe_value}</div>
            <div class="kpi-note">{safe_note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def horizontal_bar(df, x, y, title, x_label, y_label, height=470):
    chart_df = df.copy()
    chart_df["Label"] = chart_df[y].apply(short_text)

    fig = px.bar(
        chart_df.sort_values(x),
        x=x,
        y="Label",
        orientation="h",
        title=title,
        labels={x: x_label, "Label": y_label},
        text=x
    )

    fig.update_traces(
        texttemplate="%{x:,.1f}",
        textposition="outside",
        marker_line_width=0
    )

    fig.update_layout(
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font_size=20,
        title_font_color="#102A43",
        font=dict(size=13, color="#102A43"),
        xaxis=dict(showgrid=True, gridcolor="rgba(16,42,67,0.08)"),
        yaxis=dict(title=None),
        margin=dict(l=8, r=30, t=60, b=30)
    )

    return fig

def create_pdf_report(
    current_col,
    imports_total,
    exports_total,
    deficit_total,
    total_trade,
    import_export_ratio,
    top_import_product,
    top_import_value,
    top_export_product,
    top_export_value,
    top_import_country,
    top_export_country,
    top_customs_office,
    top_customs_value,
    trend_summary=None
):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TradePulseTitle",
        parent=styles["Title"],
        fontSize=24,
        leading=30,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=12
    )

    subtitle_style = ParagraphStyle(
        "TradePulseSubtitle",
        parent=styles["BodyText"],
        fontSize=11,
        leading=16,
        textColor=colors.HexColor("#475569"),
        spaceAfter=18
    )

    section_style = ParagraphStyle(
        "TradePulseSection",
        parent=styles["Heading2"],
        fontSize=15,
        leading=20,
        textColor=colors.HexColor("#102A43"),
        spaceBefore=14,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "TradePulseBody",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=8
    )

    small_style = ParagraphStyle(
        "TradePulseSmall",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#64748B")
    )

    story = []

    story.append(Paragraph("TRADEPULSE NEPAL", small_style))
    story.append(Paragraph("Monthly Trade Intelligence Brief", title_style))

    story.append(Paragraph(
        """
        A data-driven summary of Nepal's foreign trade movement based on Department of Customs data.
        This brief converts customs statistics into market signals, business opportunities, policy risks,
        and report-ready insights.
        """,
        subtitle_style
    ))

    meta_data = [
        ["Current period", current_col],
        ["Source", "Department of Customs, Government of Nepal"],
        ["Unit", "Rs. billion"],
        ["Conversion", "Source values in Rs. thousands divided by 1,000,000"]
    ]

    meta_table = Table(meta_data, colWidths=[1.55 * inch, 4.65 * inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#0F172A")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#334155")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))

    story.append(meta_table)
    story.append(Spacer(1, 18))

    story.append(Paragraph("1. Market Snapshot", section_style))

    kpi_data = [
        ["Indicator", "Value"],
        ["Total Imports", f"Rs {imports_total:,.2f} billion"],
        ["Total Exports", f"Rs {exports_total:,.2f} billion"],
        ["Trade Deficit", f"Rs {deficit_total:,.2f} billion"],
        ["Total Foreign Trade", f"Rs {total_trade:,.2f} billion"],
        ["Import / Export Ratio", f"{import_export_ratio:,.1f}x"]
    ]

    kpi_table = Table(kpi_data, colWidths=[2.6 * inch, 3.6 * inch])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#1E293B")),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))

    story.append(kpi_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        f"""
        Nepal recorded total imports of <b>Rs {imports_total:,.2f} billion</b> and exports of
        <b>Rs {exports_total:,.2f} billion</b>. Imports were around
        <b>{import_export_ratio:,.1f} times</b> larger than exports, indicating continued import dependence.
        """,
        body_style
    ))

    story.append(Paragraph("2. Product Movement", section_style))

    product_data = [
        ["Signal", "Product", "Value"],
        ["Top import product", str(top_import_product), f"Rs {top_import_value:,.2f} billion"],
        ["Top export product", str(top_export_product), f"Rs {top_export_value:,.2f} billion"]
    ]

    product_table = Table(product_data, colWidths=[1.65 * inch, 3.15 * inch, 1.4 * inch])
    product_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#102A43")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.3),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))

    story.append(product_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        f"""
        The leading imported product was <b>{top_import_product}</b>, while the leading exported product was
        <b>{top_export_product}</b>. These products can be used as starting points for product-level
        research, market mapping, sourcing analysis, and export-diversification discussion.
        """,
        body_style
    ))

    story.append(Paragraph("3. Country and Customs Route Movement", section_style))

    country_route_data = [
        ["Signal", "Result"],
        ["Largest import partner", str(top_import_country)],
        ["Largest export destination", str(top_export_country)],
        ["Largest customs route by import value", str(top_customs_office)],
        ["Import value handled by top route", f"Rs {top_customs_value:,.2f} billion"]
    ]

    country_route_table = Table(country_route_data, colWidths=[2.4 * inch, 3.8 * inch])
    country_route_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))

    story.append(country_route_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        f"""
        The largest import partner was <b>{top_import_country}</b>, while the largest export destination was
        <b>{top_export_country}</b>. The most important customs office by import value was
        <b>{top_customs_office}</b>. These indicators help identify country dependence and logistics concentration.
        """,
        body_style
    ))


    if trend_summary:
        story.append(Paragraph("4. Multi-Month Trend Movement", section_style))

        trend_data = [
            ["Signal", "Result"],
            ["Files compared", f"{trend_summary.get('files_used', 0)} monthly files"],
            ["Latest period", str(trend_summary.get('latest_period', ''))],
            ["Latest cumulative imports", f"Rs {trend_summary.get('latest_imports', 0):,.2f} billion"],
            ["Latest cumulative exports", f"Rs {trend_summary.get('latest_exports', 0):,.2f} billion"],
            ["Latest trade deficit", f"Rs {trend_summary.get('latest_deficit', 0):,.2f} billion"],
            ["Latest import change", f"Rs {trend_summary.get('import_change', 0):,.2f} billion"],
            ["Latest export change", f"Rs {trend_summary.get('export_change', 0):,.2f} billion"],
            ["Deficit change", f"Rs {trend_summary.get('deficit_change', 0):,.2f} billion"],
        ]

        trend_table = Table(trend_data, colWidths=[2.5 * inch, 3.7 * inch])
        trend_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#102A43")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9.3),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("PADDING", (0, 0), (-1, -1), 7),
        ]))

        story.append(trend_table)
        story.append(Spacer(1, 10))
        story.append(Paragraph(str(trend_summary.get("movement_note", "")), body_style))

        if trend_summary.get("top_rising_import") or trend_summary.get("top_rising_export"):
            product_signal_rows = [["Product signal", "Latest movement"]]
            if trend_summary.get("top_rising_import"):
                product_signal_rows.append([
                    "Top rising import",
                    f"{trend_summary.get('top_rising_import')} (+Rs {trend_summary.get('top_rising_import_change', 0):,.2f} billion)"
                ])
            if trend_summary.get("top_rising_export"):
                product_signal_rows.append([
                    "Top rising export",
                    f"{trend_summary.get('top_rising_export')} (+Rs {trend_summary.get('top_rising_export_change', 0):,.2f} billion)"
                ])

            product_signal_table = Table(product_signal_rows, colWidths=[2.0 * inch, 4.2 * inch])
            product_signal_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.2),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]))
            story.append(product_signal_table)
            story.append(Spacer(1, 10))

    story.append(Paragraph("5. Business Opportunity Signals", section_style))

    opportunity_points = [
        f"High-import products such as {top_import_product} may deserve deeper import-substitution research.",
        "Large import values can indicate existing domestic demand, but profitability must be checked separately.",
        f"Export products such as {top_export_product} can be studied for export expansion and destination diversification.",
        f"Major supplier countries such as {top_import_country} may reveal sourcing, distribution, and trade-finance opportunities."
    ]

    for point in opportunity_points:
        story.append(Paragraph(f"• {point}", body_style))

    story.append(Paragraph("6. Policy Risk Signals", section_style))

    risk_points = [
        f"The trade deficit remains large at Rs {deficit_total:,.2f} billion.",
        f"Imports are around {import_export_ratio:,.1f} times larger than exports.",
        "Heavy dependence on major partner countries can create external vulnerability.",
        f"Customs concentration through {top_customs_office} can create logistics and infrastructure pressure."
    ]

    for point in risk_points:
        story.append(Paragraph(f"• {point}", body_style))

    story.append(Spacer(1, 16))
    disclaimer_table = Table(
        [[Paragraph(
            """
            <b>Methodology note:</b> This report is generated from the uploaded Department of Customs workbook.
            Monetary values are converted from Rs. thousands to Rs. billion. The opportunity and risk signals
            are screening indicators for research and discussion, not investment advice.
            """,
            small_style
        )]],
        colWidths=[6.2 * inch]
    )

    disclaimer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 10),
    ]))

    story.append(disclaimer_table)

    doc.build(story)
    buffer.seek(0)
    return buffer


def clean_hscode(series):
    return series.astype(str).str.replace(".0", "", regex=False)
def remove_total_rows(df, columns):
    clean_df = df.copy()

    for col in columns:
        if col in clean_df.columns:
            clean_df = clean_df[
                ~clean_df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["total", "grand total"])
            ]

    return clean_df





# -----------------------------
# Sector and product detail helpers
# -----------------------------

def hs2_code(value):
    """Return the first two HS digits as an integer, or None when unavailable."""
    try:
        clean = str(value).replace(".0", "").strip()
        digits = "".join(ch for ch in clean if ch.isdigit())
        if len(digits) < 2:
            return None
        return int(digits[:2])
    except Exception:
        return None


HS_CHAPTER_NAMES = {
    1: "Live animals",
    2: "Meat and edible meat offal",
    3: "Fish and crustaceans",
    4: "Dairy, eggs and honey",
    5: "Other animal products",
    6: "Live trees and plants",
    7: "Vegetables",
    8: "Fruit and nuts",
    9: "Coffee, tea and spices",
    10: "Cereals",
    11: "Milling products",
    12: "Oil seeds and medicinal plants",
    13: "Gums, resins and extracts",
    14: "Vegetable plaiting materials",
    15: "Animal or vegetable fats and oils",
    16: "Prepared meat and fish",
    17: "Sugars and confectionery",
    18: "Cocoa and preparations",
    19: "Cereal, flour and bakery preparations",
    20: "Prepared vegetables and fruit",
    21: "Miscellaneous edible preparations",
    22: "Beverages",
    23: "Food industry residues and animal fodder",
    24: "Tobacco and substitutes",
    25: "Salt, sulphur, earths, stone and cement",
    26: "Ores, slag and ash",
    27: "Mineral fuels and oils",
    28: "Inorganic chemicals",
    29: "Organic chemicals",
    30: "Pharmaceutical products",
    31: "Fertilizers",
    32: "Dyes, paints and varnishes",
    33: "Essential oils, perfumes and cosmetics",
    34: "Soap, waxes and cleaning products",
    35: "Albuminoidal substances and glues",
    36: "Explosives and pyrotechnic products",
    37: "Photographic goods",
    38: "Miscellaneous chemical products",
    39: "Plastics and articles",
    40: "Rubber and articles",
    41: "Raw hides and skins",
    42: "Leather articles",
    43: "Furskins and artificial fur",
    44: "Wood and wood articles",
    45: "Cork and articles",
    46: "Basketware and wickerwork",
    47: "Pulp of wood",
    48: "Paper and paperboard",
    49: "Printed books and newspapers",
    50: "Silk",
    51: "Wool",
    52: "Cotton",
    53: "Other vegetable textile fibres",
    54: "Man-made filaments",
    55: "Man-made staple fibres",
    56: "Wadding, felt and nonwovens",
    57: "Carpets",
    58: "Special woven fabrics",
    59: "Coated textile fabrics",
    60: "Knitted or crocheted fabrics",
    61: "Knitted apparel",
    62: "Non-knitted apparel",
    63: "Other made-up textile articles",
    64: "Footwear",
    65: "Headgear",
    66: "Umbrellas and walking sticks",
    67: "Feathers and artificial flowers",
    68: "Stone, plaster, cement and asbestos articles",
    69: "Ceramic products",
    70: "Glass and glassware",
    71: "Precious stones, metals and jewellery",
    72: "Iron and steel",
    73: "Articles of iron or steel",
    74: "Copper and articles",
    75: "Nickel and articles",
    76: "Aluminium and articles",
    78: "Lead and articles",
    79: "Zinc and articles",
    80: "Tin and articles",
    81: "Other base metals",
    82: "Tools and cutlery",
    83: "Miscellaneous base metal articles",
    84: "Machinery and mechanical appliances",
    85: "Electrical machinery and electronics",
    86: "Railway or tramway equipment",
    87: "Vehicles and parts",
    88: "Aircraft and parts",
    89: "Ships and boats",
    90: "Optical, photographic and medical instruments",
    91: "Clocks and watches",
    92: "Musical instruments",
    93: "Arms and ammunition",
    94: "Furniture, bedding and lighting",
    95: "Toys, games and sports goods",
    96: "Miscellaneous manufactured articles",
    97: "Works of art and antiques",
}


def hs_chapter_label(value):
    """Return a readable HS chapter label such as '27 - Mineral fuels and oils'."""
    chapter = hs2_code(value)
    if chapter is None:
        return "Unknown HS chapter"
    return f"{chapter:02d} - {HS_CHAPTER_NAMES.get(chapter, 'Other / reserved HS chapter')}"


def sector_from_hscode(value):
    """Map HS chapters to cleaner HS-based sectors.

    This is more reliable than keyword matching because it uses the international
    HS chapter structure. It is still a broad analytical grouping, not an official
    Department of Customs sector classification.
    """
    chapter = hs2_code(value)

    if chapter is None:
        return "Other / unclassified goods"

    if 1 <= chapter <= 5:
        return "Animal products"
    if 6 <= chapter <= 14:
        return "Vegetables, cereals & farm products"
    if 15 <= chapter <= 24:
        return "Processed food, oils & beverages"
    if 25 <= chapter <= 26:
        return "Minerals & construction inputs"
    if chapter == 27:
        return "Petroleum & energy"
    if 28 <= chapter <= 29:
        return "Industrial chemicals"
    if chapter == 30:
        return "Pharmaceuticals & health"
    if chapter == 31:
        return "Fertilizers"
    if 32 <= chapter <= 38:
        return "Chemical products"
    if 39 <= chapter <= 40:
        return "Plastics & rubber"
    if 41 <= chapter <= 43:
        return "Leather & animal hides"
    if 44 <= chapter <= 49:
        return "Wood, paper & printed goods"
    if 50 <= chapter <= 60:
        return "Textile materials"
    if 61 <= chapter <= 63:
        return "Garments & made-up textiles"
    if 64 <= chapter <= 67:
        return "Footwear & accessories"
    if 68 <= chapter <= 70:
        return "Stone, cement, ceramics & glass"
    if chapter == 71:
        return "Precious metals & jewellery"
    if 72 <= chapter <= 83:
        return "Metals & metal products"
    if chapter == 84:
        return "Machinery & mechanical appliances"
    if chapter == 85:
        return "Electrical & electronics"
    if 86 <= chapter <= 89:
        return "Vehicles & transport equipment"
    if chapter == 90:
        return "Precision, optical & medical devices"
    if 91 <= chapter <= 93:
        return "Watches, instruments & special goods"
    if 94 <= chapter <= 96:
        return "Furniture, toys & miscellaneous manufactures"
    if 97 <= chapter <= 99:
        return "Art, antiques & special transactions"

    return "Other / unclassified goods"


def add_sector_column(df):
    """Add HS-based Sector and HS Chapter columns without changing the original table."""
    result = df.copy()
    if "HSCode" in result.columns:
        result["HS_Chapter"] = result["HSCode"].apply(hs2_code)
        result["HS_Chapter_Label"] = result["HSCode"].apply(hs_chapter_label)
        result["Sector"] = result["HSCode"].apply(sector_from_hscode)
    else:
        result["HS_Chapter"] = None
        result["HS_Chapter_Label"] = "Unknown HS chapter"
        result["Sector"] = "Other / unclassified goods"
    return result


def build_sector_summary(imports_df, exports_df):
    """Create sector-level import/export summary."""
    imports_sector = add_sector_column(imports_df)
    exports_sector = add_sector_column(exports_df)

    imp = (
        imports_sector
        .groupby("Sector", as_index=False)["Imports_Billion"]
        .sum()
        .rename(columns={"Imports_Billion": "Imports_Billion"})
    )

    exp = (
        exports_sector
        .groupby("Sector", as_index=False)["Exports_Billion"]
        .sum()
        .rename(columns={"Exports_Billion": "Exports_Billion"})
    )

    sector_df = imp.merge(exp, on="Sector", how="outer").fillna(0)
    sector_df["Sector_Gap_Billion"] = sector_df["Imports_Billion"] - sector_df["Exports_Billion"]
    sector_df["Total_Trade_Billion"] = sector_df["Imports_Billion"] + sector_df["Exports_Billion"]

    total_imports = sector_df["Imports_Billion"].sum()
    total_exports = sector_df["Exports_Billion"].sum()

    sector_df["Import_Share"] = (
        sector_df["Imports_Billion"] / total_imports * 100 if total_imports else 0
    )
    sector_df["Export_Share"] = (
        sector_df["Exports_Billion"] / total_exports * 100 if total_exports else 0
    )
    sector_df["Import_Export_Ratio"] = sector_df.apply(
        lambda row: row["Imports_Billion"] / row["Exports_Billion"] if row["Exports_Billion"] else None,
        axis=1
    )

    return sector_df.sort_values("Imports_Billion", ascending=False)


def get_sector_top_products(imports_df, exports_df, sector_name, top_n=8):
    imports_sector = add_sector_column(imports_df)
    exports_sector = add_sector_column(exports_df)

    top_import_products = (
        imports_sector[imports_sector["Sector"] == sector_name]
        .sort_values("Imports_Billion", ascending=False)
        .head(top_n)
    )

    top_export_products = (
        exports_sector[exports_sector["Sector"] == sector_name]
        .sort_values("Exports_Billion", ascending=False)
        .head(top_n)
    )

    return top_import_products, top_export_products


def sector_signal(row):
    """Write a simple analyst note for a sector."""
    imports_value = float(row.get("Imports_Billion", 0) or 0)
    exports_value = float(row.get("Exports_Billion", 0) or 0)
    gap_value = float(row.get("Sector_Gap_Billion", 0) or 0)

    if imports_value > exports_value * 5 and imports_value >= 10:
        return "High import dependence. Useful for import-substitution research, sourcing analysis, and price-risk monitoring."
    if exports_value > imports_value and exports_value >= 1:
        return "Export-oriented sector. Useful for competitiveness and market-access analysis."
    if gap_value > 0:
        return "Net import sector. Watch domestic demand, exchange-rate pressure, and supply-chain exposure."
    return "Balanced or small sector. Interpret together with product-level details."


def product_detail_summary(product_row, direction="import"):
    """Create a compact product summary from a selected row."""
    description = product_row.get("Description", "Selected product")
    hs = product_row.get("HSCode", "")
    sector = sector_from_hscode(hs)

    if direction == "import":
        value = product_row.get("Imports_Billion", 0)
        unit = product_row.get("Unit", "")
        quantity = product_row.get("Quantity", "")
        return {
            "description": description,
            "hs": hs,
            "sector": sector,
            "value_label": "Import value",
            "value": value,
            "unit": unit,
            "quantity": quantity,
            "note": "High import value can signal strong domestic demand, dependency risk, or import-substitution potential."
        }

    value = product_row.get("Exports_Billion", 0)
    unit = product_row.get("Unit", "")
    quantity = product_row.get("Quantity", "")
    return {
        "description": description,
        "hs": hs,
        "sector": sector,
        "value_label": "Export value",
        "value": value,
        "unit": unit,
        "quantity": quantity,
        "note": "Export value can signal existing competitiveness, market access, or product specialization."
    }


def product_partner_table(partner_df, hscode, value_col, display_value_col, top_n=10):
    """Return top partner countries for a product from commodity-partner sheets."""
    if partner_df is None or len(partner_df) == 0 or "HSCode" not in partner_df.columns:
        return pd.DataFrame()

    temp = partner_df.copy()
    temp["HSCode_Clean"] = clean_hscode(temp["HSCode"])

    selected = temp[temp["HSCode_Clean"].astype(str) == str(hscode)].copy()

    if len(selected) == 0:
        return pd.DataFrame()

    selected = selected.sort_values(value_col, ascending=False).head(top_n)

    cols = ["Partner Countries", display_value_col]
    available_cols = [col for col in cols if col in selected.columns]

    return selected[available_cols]


def product_monthly_movement(monthly_data_path, hscode, direction="import", end_file_name=None):
    """Return monthly movement for one product from cumulative monthly files when available."""
    try:
        monthly_data_path = Path(monthly_data_path)
        if not monthly_data_path.exists():
            return pd.DataFrame()

        trend_files = get_trend_files_for_period(monthly_data_path, end_file_name)
        if len(trend_files) < 2:
            return pd.DataFrame()

        if direction == "import":
            movement = build_product_movement_table(
                trend_files,
                "5_Imports_By_Commodity",
                "Imports_Value"
            )
        else:
            movement = build_product_movement_table(
                trend_files,
                "7_Exports_By_Commodity",
                "Exports_Value"
            )

        selected = movement[movement["HSCode"].astype(str) == str(hscode)].copy()
        return selected

    except Exception:
        return pd.DataFrame()

# -----------------------------
# Automated insight engine
# -----------------------------

def fmt_money(value):
    try:
        return f"Rs {float(value):,.2f}B"
    except Exception:
        return "Not available"


def fmt_pct(value):
    try:
        return f"{float(value):,.1f}%"
    except Exception:
        return "Not available"


def clean_display(value):
    return html.escape(str(value))


def trend_monthly_signal(monthly_data_path, end_file_name=None):
    """Create a small trend signal from cumulative monthly customs files."""
    try:
        if not monthly_data_path.exists():
            return None

        trend_files = get_trend_files_for_period(monthly_data_path, end_file_name)
        if len(trend_files) < 2:
            return None

        rows = []
        for file_path in trend_files:
            try:
                rows.append(extract_trade_snapshot_from_file(file_path, file_path.name))
            except Exception:
                continue

        if len(rows) < 2:
            return None

        trend_df = pd.DataFrame(rows).sort_values("File").reset_index(drop=True)

        latest = trend_df.iloc[-1]
        previous = trend_df.iloc[-2]

        monthly_import_change = latest["Imports_Billion"] - previous["Imports_Billion"]
        monthly_export_change = latest["Exports_Billion"] - previous["Exports_Billion"]
        monthly_deficit_change = latest["Trade_Deficit_Billion"] - previous["Trade_Deficit_Billion"]

        return {
            "latest_period": latest["Period"],
            "previous_period": previous["Period"],
            "monthly_import_change": monthly_import_change,
            "monthly_export_change": monthly_export_change,
            "monthly_deficit_change": monthly_deficit_change,
            "direction": "widened" if monthly_deficit_change > 0 else "narrowed"
        }

    except Exception:
        return None


def build_automated_insights(
    imports_total,
    exports_total,
    deficit_total,
    total_trade,
    import_export_ratio,
    top_import_product,
    top_import_value,
    top_export_product,
    top_export_value,
    top_import_country,
    top_export_country,
    top_customs_office,
    top_customs_value,
    top5_route_share,
    imports,
    exports,
    countries,
    monthly_data_path
):
    """Rule-based trade analyst. No external API is used."""
    insights = {}

    deficit_share = (deficit_total / total_trade * 100) if total_trade else 0
    top_import_share = (top_import_value / imports_total * 100) if imports_total else 0
    top_export_share = (top_export_value / exports_total * 100) if exports_total else 0

    trend_signal = trend_monthly_signal(monthly_data_path, selected_trend_end_name)

    # Product concentration
    top5_import_share = 0
    top5_export_share = 0
    try:
        top5_import_share = imports.sort_values("Imports_Billion", ascending=False).head(5)["Imports_Billion"].sum() / imports_total * 100
    except Exception:
        top5_import_share = 0
    try:
        top5_export_share = exports.sort_values("Exports_Billion", ascending=False).head(5)["Exports_Billion"].sum() / exports_total * 100
    except Exception:
        top5_export_share = 0

    insights["cards"] = [
        {
            "badge": "External position",
            "title": "Imports still dominate the trade structure",
            "body": (
                f"Imports are around {import_export_ratio:,.1f} times larger than exports. "
                f"The trade deficit equals about {deficit_share:,.1f}% of total foreign trade in the selected period."
            )
        },
        {
            "badge": "Product signal",
            "title": "Top products deserve deeper tracking",
            "body": (
                f"The leading import item is {top_import_product}, worth {fmt_money(top_import_value)} "
                f"or about {top_import_share:,.1f}% of total imports. The leading export item is {top_export_product}, "
                f"worth {fmt_money(top_export_value)}."
            )
        },
        {
            "badge": "Market concentration",
            "title": "Partner dependence remains important",
            "body": (
                f"The biggest import partner is {top_import_country}, while the biggest export destination is {top_export_country}. "
                "This is useful for country-risk, sourcing, and market-access analysis."
            )
        },
        {
            "badge": "Logistics",
            "title": "Customs route pressure is visible",
            "body": (
                f"The largest customs route is {top_customs_office}, handling around {fmt_money(top_customs_value)} in imports. "
                f"The top five routes together account for about {top5_route_share:,.1f}% of import value."
            )
        },
    ]

    if trend_signal:
        insights["cards"].insert(1, {
            "badge": "Latest movement",
            "title": f"Trade deficit {trend_signal['direction']} in the latest monthly update",
            "body": (
                f"From {trend_signal['previous_period']} to {trend_signal['latest_period']}, cumulative imports changed by "
                f"{fmt_money(trend_signal['monthly_import_change'])}, exports changed by {fmt_money(trend_signal['monthly_export_change'])}, "
                f"and the trade deficit changed by {fmt_money(trend_signal['monthly_deficit_change'])}."
            )
        })

    insights["opportunities"] = [
        f"Study high-import products such as {top_import_product} for import-substitution, distribution, and sourcing opportunities.",
        f"Track export leaders such as {top_export_product} to understand where Nepal already has export momentum.",
        f"Use partner-country data around {top_import_country} and {top_export_country} for market-entry and trade-finance research.",
        "Use customs route data to identify logistics concentration and border-point dependence."
    ]

    insights["risks"] = [
        f"The import-export gap remains large: imports are {import_export_ratio:,.1f} times exports.",
        f"Top import products and routes may create concentration risk if supply chains are disrupted.",
        f"Large dependence on major partner countries can expose Nepal to external price, policy, and logistics shocks.",
        "Some opportunity signals are screening indicators only; product-level feasibility needs separate research."
    ]

    trend_sentence = ""
    if trend_signal:
        trend_sentence = (
            f"\nLatest monthly movement: from {trend_signal['previous_period']} to {trend_signal['latest_period']}, "
            f"the trade deficit {trend_signal['direction']} by {fmt_money(abs(trend_signal['monthly_deficit_change']))}."
        )

    insights["brief_text"] = f"""TradePulse Nepal — Automated Trade Analyst Brief

Current period: {current_period_display}
Source: Department of Customs monthly foreign trade data
Unit: Rs. billion, converted from source values in Rs. thousands

1. Market Snapshot
Nepal recorded imports of {fmt_money(imports_total)} and exports of {fmt_money(exports_total)}.
The trade deficit stood at {fmt_money(deficit_total)}, with imports around {import_export_ratio:,.1f} times larger than exports.
The deficit is equal to about {deficit_share:,.1f}% of total foreign trade.{trend_sentence}

2. Product Signal
The leading imported product was {top_import_product}, worth {fmt_money(top_import_value)}.
The leading exported product was {top_export_product}, worth {fmt_money(top_export_value)}.
The top five import products represent about {top5_import_share:,.1f}% of total imports, while the top five export products represent about {top5_export_share:,.1f}% of total exports.

3. Country and Route Signal
The largest import partner was {top_import_country}.
The largest export destination was {top_export_country}.
The most important customs office by import value was {top_customs_office}, handling around {fmt_money(top_customs_value)}.

4. Analyst Read
The data shows Nepal's continued import dependence and a large trade deficit. The most useful next analysis is to track which products are rising month by month, whether export growth is broad-based or concentrated, and where high-import products create practical business or policy research opportunities.
"""

    return insights




def ask_tradepulse_rule_based(
    question,
    imports_total,
    exports_total,
    deficit_total,
    total_trade,
    import_export_ratio,
    top_import_product,
    top_import_value,
    top_export_product,
    top_export_value,
    top_import_country,
    top_export_country,
    top_customs_office,
    top_customs_value,
    top5_route_share,
    imports,
    exports,
    countries,
    customs,
    monthly_data_path
):
    """Free rule-based Ask TradePulse assistant. No paid AI API is used."""
    q = str(question or "").lower().strip()

    if not q:
        return "Ask a question about imports, exports, trade deficit, products, countries, customs routes, opportunities, or risks."

    trend_summary = build_trend_summary_for_report(monthly_data_path, selected_trend_end_name)
    deficit_share = (deficit_total / total_trade * 100) if total_trade else 0

    top5_imports = imports.sort_values("Imports_Billion", ascending=False).head(5)
    top5_exports = exports.sort_values("Exports_Billion", ascending=False).head(5)
    top5_import_countries = countries.sort_values("Imports_Billion", ascending=False).head(5)
    top5_export_countries = countries.sort_values("Exports_Billion", ascending=False).head(5)
    top5_customs = customs.sort_values("Imports_Billion", ascending=False).head(5)

    def product_lines(df, value_col):
        lines = []
        for idx, row in enumerate(df.itertuples(index=False), start=1):
            description = getattr(row, "Description", "Unknown")
            value = getattr(row, value_col, 0)
            lines.append(f"{idx}. {description} — {fmt_money(value)}")
        return "\n".join(lines)

    def country_lines(df, value_col):
        lines = []
        for idx, row in enumerate(df.itertuples(index=False), start=1):
            country = getattr(row, "Partner Countries", "Unknown")
            value = getattr(row, value_col, 0)
            lines.append(f"{idx}. {country} — {fmt_money(value)}")
        return "\n".join(lines)

    def route_lines(df):
        lines = []
        for idx, row in enumerate(df.itertuples(index=False), start=1):
            route = getattr(row, "Customs", "Unknown")
            value = getattr(row, "Imports_Billion", 0)
            lines.append(f"{idx}. {route} — {fmt_money(value)}")
        return "\n".join(lines)

    def analyst_answer(title, direct, why=None, signal=None, caution=None):
        parts = [f"### {title}", f"**Direct answer**\n\n{direct}"]
        if why:
            parts.append(f"**Why it matters**\n\n{why}")
        if signal:
            parts.append(f"**Data signal**\n\n{signal}")
        if caution:
            parts.append(f"**Caution**\n\n{caution}")
        return "\n\n".join(parts)

    if any(word in q for word in ["changed", "latest", "movement", "month", "monthly", "jestha", "baisakh"]):
        if trend_summary:
            return analyst_answer(
                "Latest movement",
                f"From **{trend_summary['previous_period']}** to **{trend_summary['latest_period']}**, cumulative imports changed by **{fmt_money(trend_summary['import_change'])}**, exports changed by **{fmt_money(trend_summary['export_change'])}**, and the trade deficit changed by **{fmt_money(trend_summary['deficit_change'])}**.",
                "This shows whether the latest month added more pressure through imports, helped through exports, or changed the external balance.",
                trend_summary['movement_note'],
                "This is based on cumulative monthly Customs Excel files in the monthly_data folder."
            )
        return "Monthly trend data is not available yet. Add monthly Excel files inside the `monthly_data` folder to enable latest movement answers."

    if any(word in q for word in ["increase", "increased", "rising", "rose", "growth", "grew"]):
        if trend_summary and trend_summary.get("top_rising_import"):
            answer = (
                f"**Strongest product movement**\n\n"
                f"The strongest import increase in the latest monthly movement came from **{trend_summary['top_rising_import']}**, "
                f"with a change of **{fmt_money(trend_summary['top_rising_import_change'])}**.\n\n"
            )
            if trend_summary.get("top_rising_export"):
                answer += (
                    f"The strongest export increase came from **{trend_summary['top_rising_export']}**, "
                    f"with a change of **{fmt_money(trend_summary['top_rising_export_change'])}**."
                )
            return analyst_answer(
                "Strongest product movement",
                answer.replace("**Strongest product movement**\n\n", ""),
                "Product movement helps identify where demand, prices, sourcing, or export momentum may be changing fastest.",
                "Use this as a first screening signal before checking price, volume, seasonality, and policy context.",
                "This answer uses month-to-month changes from cumulative files, so unusual one-off entries should be checked manually."
            )
        return "I could not detect product-level monthly movement yet. Check that monthly Excel files are available inside `monthly_data`."

    if "deficit" in q or "gap" in q:
        return analyst_answer(
            "Trade deficit",
            f"The trade deficit in the selected period is **{fmt_money(deficit_total)}**.",
            "A large deficit means imports are much higher than exports, which can create pressure on foreign exchange, domestic production, and external stability.",
            f"Imports are around **{import_export_ratio:,.1f} times** larger than exports. The deficit equals about **{deficit_share:,.1f}%** of total foreign trade.",
            "This is a descriptive dashboard signal, not a complete macroeconomic diagnosis."
        )

    if "ratio" in q or "import export" in q or "import/export" in q:
        return analyst_answer(
            "Import-export ratio",
            f"Imports are around **{import_export_ratio:,.1f} times** larger than exports in the selected period.",
            "This gives a quick view of how import-dominated Nepal's trade structure is.",
            f"Imports: **{fmt_money(imports_total)}**. Exports: **{fmt_money(exports_total)}**.",
            "Use the ratio with product and country details for a fuller interpretation."
        )

    if ("import" in q and "product" in q) or "importing most" in q or "import most" in q or "nepal importing" in q or "what is nepal importing" in q:
        return analyst_answer(
            "Top import products",
            f"The leading import product is **{top_import_product}**, worth **{fmt_money(top_import_value)}**.",
            "Large import products are useful for studying domestic demand, supply dependence, distribution markets, and possible import-substitution areas.",
            product_lines(top5_imports, 'Imports_Billion'),
            "High import value alone does not mean an easy business opportunity; check margins, regulation, supply chains, and competition."
        )

    if ("export" in q and "product" in q) or "exporting most" in q or "export most" in q or "nepal exporting" in q or "what is nepal exporting" in q:
        return analyst_answer(
            "Top export products",
            f"The leading export product is **{top_export_product}**, worth **{fmt_money(top_export_value)}**.",
            "Export leaders show where Nepal already has some market traction, but concentration can also create vulnerability.",
            product_lines(top5_exports, 'Exports_Billion'),
            "Export value should be checked with destination markets, product margins, and policy incentives."
        )


    if "sector" in q or "category" in q or "industries" in q or "industry" in q:
        try:
            temp_sector = build_sector_summary(imports, exports)
            if temp_sector is not None and len(temp_sector) > 0:
                top_import_sector = temp_sector.sort_values("Imports_Billion", ascending=False).iloc[0]
                top_export_sector = temp_sector.sort_values("Exports_Billion", ascending=False).iloc[0]
                top_gap_sector = temp_sector.sort_values("Sector_Gap_Billion", ascending=False).iloc[0]
                sector_lines = []
                for idx, row in enumerate(temp_sector.sort_values("Imports_Billion", ascending=False).head(6).itertuples(index=False), start=1):
                    sector_lines.append(
                        f"{idx}. {getattr(row, 'Sector', 'Unknown')} — imports {fmt_money(getattr(row, 'Imports_Billion', 0))}, exports {fmt_money(getattr(row, 'Exports_Billion', 0))}, gap {fmt_money(getattr(row, 'Sector_Gap_Billion', 0))}"
                    )
                return analyst_answer(
                    "Sector read",
                    f"The largest import sector is **{top_import_sector['Sector']}**, while the largest export sector is **{top_export_sector['Sector']}**. The largest sector trade gap is in **{top_gap_sector['Sector']}**.",
                    "Sector grouping helps move from thousands of product lines to a cleaner view of where Nepal's trade dependence and export strengths are concentrated.",
                    "\n".join(sector_lines),
                    "Sectors are grouped using HS chapters. They are analytical groups, not official Department of Customs sector categories."
                )
        except Exception:
            pass

    if "detail" in q or "profile" in q or "tell me about" in q or "about" in q or "hs code" in q or "hscode" in q:
        try:
            stopwords = {
                "what", "about", "detail", "profile", "tell", "nepal", "import", "imports", "export", "exports",
                "product", "products", "trade", "most", "main", "top", "which", "where", "this", "that", "give", "explain"
            }
            tokens = [token for token in q.replace("?", " ").replace(",", " ").split() if len(token) >= 4 and token not in stopwords]
            digits = "".join(ch for ch in q if ch.isdigit())
            pool_parts = []
            imp_pool = imports.copy()
            imp_pool["Direction"] = "Import"
            imp_pool["Value_Billion"] = imp_pool.get("Imports_Billion", 0)
            exp_pool = exports.copy()
            exp_pool["Direction"] = "Export"
            exp_pool["Value_Billion"] = exp_pool.get("Exports_Billion", 0)
            pool = pd.concat([imp_pool, exp_pool], ignore_index=True, sort=False)
            pool["Description_Lower"] = pool["Description"].astype(str).str.lower()
            pool["HSCode_Text"] = pool["HSCode"].astype(str)

            matched = pool.iloc[0:0].copy()
            if len(digits) >= 2:
                matched = pool[pool["HSCode_Text"].str.startswith(digits[:2])].copy()
            if len(matched) == 0 and tokens:
                mask = False
                for token in tokens:
                    mask = mask | pool["Description_Lower"].str.contains(token, na=False, regex=False)
                matched = pool[mask].copy()

            if len(matched) > 0:
                best = matched.sort_values("Value_Billion", ascending=False).iloc[0]
                desc = best.get("Description", "Selected product")
                hs = best.get("HSCode", "")
                direction = best.get("Direction", "Trade")
                value = best.get("Value_Billion", 0)
                sector = sector_from_hscode(hs)
                return analyst_answer(
                    "Product profile",
                    f"The closest product match is **{desc}** under HS code **{hs}**. In the current dashboard it appears mainly as **{direction.lower()} trade**, valued at **{fmt_money(value)}**.",
                    "A product profile helps connect the product's value with its sector, possible dependence, and deeper questions about countries and routes.",
                    f"Sector: **{sector}**. Direction: **{direction}**. Value: **{fmt_money(value)}**.",
                    "This is a quick match from product descriptions/HS codes. Use the Product Detail tab for more precise product-level partner and movement data."
                )
        except Exception:
            pass

    if "country" in q or "partner" in q or "india" in q or "china" in q:
        if "export" in q:
            return analyst_answer(
                "Export destinations",
                f"The biggest export destination is **{top_export_country}**.",
                "Destination concentration matters because exports can become vulnerable to demand, policy, or border changes in one market.",
                country_lines(top5_export_countries, 'Exports_Billion'),
                "This shows value concentration, not profitability or competitiveness."
            )
        return analyst_answer(
            "Import partners",
            f"The biggest import partner is **{top_import_country}**.",
            "Partner concentration shows where Nepal's sourcing dependence is strongest.",
            country_lines(top5_import_countries, 'Imports_Billion'),
            "This is based on Customs value data and does not explain the reason for dependence by itself."
        )

    if "customs" in q or "route" in q or "border" in q or "birgunj" in q or "handles most trade" in q or "office handles" in q:
        return analyst_answer(
            "Customs route signal",
            f"The largest customs route by import value is **{top_customs_office}**, handling around **{fmt_money(top_customs_value)}**.",
            "Route concentration matters for logistics, border pressure, transport planning, and supply-chain risk.",
            f"The top five routes together account for about **{top5_route_share:,.1f}%** of import value.\n\n{route_lines(top5_customs)}",
            "Customs value does not directly measure congestion, delay, or transport cost."
        )

    if "opportunity" in q or "business" in q or "substitution" in q or "market" in q:
        return analyst_answer(
            "Opportunity read",
            f"The clearest opportunity areas are high-import products such as **{top_import_product}**, export leaders such as **{top_export_product}**, partner dependence around **{top_import_country}**, and logistics concentration around **{top_customs_office}**.",
            "High import value can point to strong domestic demand or sourcing dependence. Export leaders can point to existing market traction. Route and country concentration can reveal logistics or diversification opportunities.",
            f"Top import product: **{top_import_product}** ({fmt_money(top_import_value)}). Top export product: **{top_export_product}** ({fmt_money(top_export_value)}). Biggest import partner: **{top_import_country}**. Main customs route: **{top_customs_office}**.",
            "This is only a screening signal, not investment advice. A real business decision needs price, volume, regulation, competition, and field validation."
        )

    if "risk" in q or "vulnerable" in q or "vulnerability" in q or "dependence" in q or "dependent" in q:
        return analyst_answer(
            "Risk read",
            f"The main risk is Nepal's large import dependence: imports are **{import_export_ratio:,.1f} times** exports.",
            "Heavy import dependence can expose the economy to foreign exchange pressure, global price shocks, border disruptions, and partner-country policy changes.",
            f"Trade deficit: **{fmt_money(deficit_total)}**. Biggest import partner: **{top_import_country}**. Largest customs route: **{top_customs_office}**.",
            "This is a risk-screening summary, not a full macroeconomic stress test."
        )

    if "media" in q or "story" in q or "article" in q or "headline" in q or "news" in q:
        return analyst_answer(
            "Media story ideas",
            f"A strong story angle is: **Nepal’s trade gap remains large as imports are led by {top_import_product}, while exports remain concentrated around {top_export_product}.**",
            "This is useful for journalists, students, and researchers because it turns raw Customs data into a clear economic story.",
            f"Possible angles:\n1. Import dependence around **{top_import_product}**.\n2. Export concentration around **{top_export_product}**.\n3. Partner dependence around **{top_import_country}**.\n4. Route concentration around **{top_customs_office}**.\n5. The wider trade deficit of **{fmt_money(deficit_total)}**.",
            "This gives story ideas only. Before publishing, verify product details, policy context, prices, and expert comments."
        )

    if "brief" in q or "report" in q or "note" in q:
        return analyst_answer(
            "Short trade brief",
            f"Nepal recorded imports of **{fmt_money(imports_total)}**, exports of **{fmt_money(exports_total)}**, and a trade deficit of **{fmt_money(deficit_total)}** in the selected Customs dataset.",
            "The main analytical message is that imports continue to dominate the trade structure, while exports remain much smaller and concentrated in selected products.",
            f"Leading import: **{top_import_product}**. Leading export: **{top_export_product}**. Biggest import partner: **{top_import_country}**. Main customs route: **{top_customs_office}**. Import-export ratio: **{import_export_ratio:,.1f}x**.",
            "This is an automated brief generated from dashboard data. It should be reviewed before use in formal reports or media articles."
        )

    if "summary" in q or "explain" in q or "overview" in q or "so what" in q:
        return analyst_answer(
            "TradePulse summary",
            f"Nepal's selected-period trade data shows imports of **{fmt_money(imports_total)}**, exports of **{fmt_money(exports_total)}**, and a trade deficit of **{fmt_money(deficit_total)}**.",
            "The main story is that imports still dominate exports, while product, country, and customs-route concentration reveal where deeper analysis should focus.",
            f"Import-export ratio: **{import_export_ratio:,.1f}x**. Leading import: **{top_import_product}**. Leading export: **{top_export_product}**. Biggest import partner: **{top_import_country}**. Main customs route: **{top_customs_office}**.",
            "This summary is generated from the selected Customs dataset loaded in the dashboard."
        )

    return (
        f"I can answer this from the dashboard data, but I may need a more specific question.\n\n"
        f"Try asking: **What changed in the latest month?**, **Which product increased the most?**, "
        f"**What are the top import products?**, **Which country dominates imports?**, "
        f"**What is the trade deficit?**, or **What are the main risks?**"
    )



def build_tradepulse_ai_context(
    current_col,
    source_file_label,
    latest_month_label,
    monthly_files_count,
    imports_total,
    exports_total,
    deficit_total,
    total_trade,
    import_export_ratio,
    imports,
    exports,
    countries,
    customs,
    sector_summary,
    monthly_data_path
):
    """Create a compact, processed-data-only context for an LLM."""
    trend_summary = build_trend_summary_for_report(monthly_data_path, selected_trend_end_name)

    def top_lines(df, name_col, value_col, n=8):
        if df is None or len(df) == 0 or value_col not in df.columns:
            return "Not available"
        temp = df.sort_values(value_col, ascending=False).head(n)
        lines = []
        for idx, row in enumerate(temp.itertuples(index=False), start=1):
            name = getattr(row, name_col, "Unknown") if hasattr(row, name_col) else row._asdict().get(name_col, "Unknown")
            value = getattr(row, value_col, 0)
            lines.append(f"{idx}. {name}: {fmt_money(value)}")
        return "\n".join(lines)

    top_import_products = top_lines(imports, "Description", "Imports_Billion")
    top_export_products = top_lines(exports, "Description", "Exports_Billion")
    top_import_countries = top_lines(countries, "Partner Countries", "Imports_Billion")
    top_export_countries = top_lines(countries, "Partner Countries", "Exports_Billion")
    top_routes = top_lines(customs, "Customs", "Imports_Billion")

    if sector_summary is not None and len(sector_summary) > 0:
        sector_lines = []
        for idx, row in enumerate(sector_summary.sort_values("Imports_Billion", ascending=False).head(8).itertuples(index=False), start=1):
            sector_lines.append(
                f"{idx}. {getattr(row, 'Sector', 'Unknown')}: imports {fmt_money(getattr(row, 'Imports_Billion', 0))}, exports {fmt_money(getattr(row, 'Exports_Billion', 0))}, gap {fmt_money(getattr(row, 'Sector_Gap_Billion', 0))}"
            )
        sector_text = "\n".join(sector_lines)
    else:
        sector_text = "Not available"

    trend_text = "Monthly movement not available."
    if trend_summary:
        trend_text = (
            f"Latest movement from {trend_summary['previous_period']} to {trend_summary['latest_period']}: "
            f"imports changed by {fmt_money(trend_summary['import_change'])}, "
            f"exports changed by {fmt_money(trend_summary['export_change'])}, "
            f"deficit changed by {fmt_money(trend_summary['deficit_change'])}. "
            f"{trend_summary['movement_note']}"
        )

    return f"""
TradePulse Nepal processed data context
Data source: Department of Customs, Government of Nepal
Main dashboard file: {source_file_label}
Current period/column: {current_period_display}
Latest monthly file: {latest_month_label}
Monthly files loaded: {monthly_files_count}
Unit: Rs. billion, converted from source values in Rs. thousands

Market snapshot:
- Imports: {fmt_money(imports_total)}
- Exports: {fmt_money(exports_total)}
- Trade deficit: {fmt_money(deficit_total)}
- Total foreign trade: {fmt_money(total_trade)}
- Import-export ratio: {import_export_ratio:,.1f}x

Latest monthly signal:
{trend_text}

Top import products:
{top_import_products}

Top export products:
{top_export_products}

Top import partners:
{top_import_countries}

Top export destinations:
{top_export_countries}

Top customs routes by import value:
{top_routes}

Sector summary:
{sector_text}
""".strip()


def build_compact_gemini_context(
    current_col,
    source_file_label,
    latest_month_label,
    monthly_files_count,
    imports_total,
    exports_total,
    deficit_total,
    total_trade,
    import_export_ratio,
    imports,
    exports,
    countries,
    customs,
    sector_summary=None,
    monthly_data_path=None,
    context_mode="Detailed",
    trend_end_file_name=None,
):
    """Build a processed TradePulse context for Gemini.

    This is still safer than sending raw Excel files, but it gives Gemini much more useful
    information: product rankings, sector rankings, partner countries, customs routes,
    monthly aggregate movement, and product-level rising/falling signals.
    """

    mode = str(context_mode or "Detailed").lower()
    if "light" in mode:
        top_n = 8
        movement_n = 8
        cap = 9000
    elif "max" in mode or "full" in mode or "detail" in mode:
        top_n = 30
        movement_n = 25
        cap = 24000
    else:
        top_n = 15
        movement_n = 12
        cap = 14000

    def clean_label(value):
        label = str(value).strip()
        if not label or label.lower() in ["nan", "total", "grand total", "none"]:
            return "Unknown"
        return label

    def top_rows_text(df, name_col, value_col, n=10, extra_cols=None):
        try:
            if df is None or df.empty or name_col not in df.columns or value_col not in df.columns:
                return "Not available"
            temp = df.copy()
            temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce").fillna(0)
            temp = temp.sort_values(value_col, ascending=False).head(n)
            lines = []
            for idx, (_, row) in enumerate(temp.iterrows(), start=1):
                label = clean_label(row.get(name_col, "Unknown"))
                value = safe_number(row.get(value_col, 0))
                extras = []
                if extra_cols:
                    for c in extra_cols:
                        if c in temp.columns:
                            extras.append(f"{c}: {row.get(c)}")
                extra_text = f" | {'; '.join(extras)}" if extras else ""
                if label != "Unknown":
                    lines.append(f"{idx}. {label}: {fmt_money(value)}{extra_text}")
            return "\n".join(lines) if lines else "Not available"
        except Exception:
            return "Not available"

    def sector_rows_text(df, n=15):
        try:
            if df is None or df.empty or "Sector" not in df.columns:
                return "Not available"
            temp = df.copy()
            sort_col = "Sector_Total_Trade_Billion" if "Sector_Total_Trade_Billion" in temp.columns else (
                "Imports_Billion" if "Imports_Billion" in temp.columns else None
            )
            if not sort_col:
                return "Not available"
            temp[sort_col] = pd.to_numeric(temp[sort_col], errors="coerce").fillna(0)
            temp = temp.sort_values(sort_col, ascending=False).head(n)
            lines = []
            for idx, (_, row) in enumerate(temp.iterrows(), start=1):
                sector = clean_label(row.get("Sector", "Unknown"))
                imp = safe_number(row.get("Imports_Billion", 0))
                exp = safe_number(row.get("Exports_Billion", 0))
                gap = safe_number(row.get("Sector_Gap_Billion", imp - exp))
                total = safe_number(row.get("Sector_Total_Trade_Billion", imp + exp))
                lines.append(
                    f"{idx}. {sector}: total {fmt_money(total)}, imports {fmt_money(imp)}, exports {fmt_money(exp)}, gap {fmt_money(gap)}"
                )
            return "\n".join(lines) if lines else "Not available"
        except Exception:
            return "Not available"

    def product_movement_text(sheet_name, value_col, label, n=15):
        try:
            if not monthly_data_path:
                return f"{label} movement not available."
            trend_files = get_trend_files_for_period(monthly_data_path, trend_end_file_name)
            if len(trend_files) < 2:
                return f"{label} movement not available."
            movement = build_product_movement_table(trend_files, sheet_name, value_col)
            movement = movement.copy()
            movement["Latest_Month_Billion"] = pd.to_numeric(movement["Latest_Month_Billion"], errors="coerce").fillna(0)
            movement["Change_Billion"] = pd.to_numeric(movement["Change_Billion"], errors="coerce").fillna(0)
            # Keep economically meaningful products; avoids tiny/noisy changes.
            meaningful = movement[movement["Latest_Month_Billion"].abs() >= 0.01].copy()
            if meaningful.empty:
                meaningful = movement.copy()

            rising = meaningful.sort_values("Change_Billion", ascending=False).head(n)
            falling = meaningful.sort_values("Change_Billion", ascending=True).head(n)

            def rows_to_lines(df, title):
                lines = [title]
                for idx, (_, row) in enumerate(df.iterrows(), start=1):
                    desc = clean_label(row.get("Description", "Unknown"))
                    hs = str(row.get("HSCode", "")).strip()
                    latest = safe_number(row.get("Latest_Month_Billion", 0))
                    change = safe_number(row.get("Change_Billion", 0))
                    prev_period = row.get("Previous_Period", "previous")
                    latest_period = row.get("Latest_Period", "latest")
                    lines.append(
                        f"{idx}. {desc} (HS {hs}): latest month {fmt_money(latest)}, change {fmt_money(change)} from {prev_period} to {latest_period}"
                    )
                return "\n".join(lines)

            return rows_to_lines(rising, f"Top rising {label} products:") + "\n\n" + rows_to_lines(falling, f"Top falling {label} products:")
        except Exception as exc:
            return f"{label} movement not available."

    def monthly_trend_table_text(n=20):
        try:
            if not monthly_data_path:
                return "Monthly trend table not available."
            trend_files = get_trend_files_for_period(monthly_data_path, trend_end_file_name)
            rows = []
            for file_path in trend_files:
                try:
                    rows.append(extract_trade_snapshot_from_file(file_path, file_path.name))
                except Exception:
                    continue
            if not rows:
                return "Monthly trend table not available."
            trend_df = pd.DataFrame(rows)
            trend_df["Monthly_Imports_Billion"] = trend_df["Imports_Billion"].diff()
            trend_df["Monthly_Exports_Billion"] = trend_df["Exports_Billion"].diff()
            trend_df["Monthly_Deficit_Billion"] = trend_df["Trade_Deficit_Billion"].diff()
            trend_df.loc[trend_df.index[0], "Monthly_Imports_Billion"] = trend_df.loc[trend_df.index[0], "Imports_Billion"]
            trend_df.loc[trend_df.index[0], "Monthly_Exports_Billion"] = trend_df.loc[trend_df.index[0], "Exports_Billion"]
            trend_df.loc[trend_df.index[0], "Monthly_Deficit_Billion"] = trend_df.loc[trend_df.index[0], "Trade_Deficit_Billion"]
            lines = []
            for _, row in trend_df.tail(n).iterrows():
                lines.append(
                    f"- {row['Period']}: cumulative imports {fmt_money(row['Imports_Billion'])}, exports {fmt_money(row['Exports_Billion'])}, deficit {fmt_money(row['Trade_Deficit_Billion'])}; monthly imports {fmt_money(row['Monthly_Imports_Billion'])}, exports {fmt_money(row['Monthly_Exports_Billion'])}, deficit {fmt_money(row['Monthly_Deficit_Billion'])}"
                )
            return "\n".join(lines)
        except Exception:
            return "Monthly trend table not available."

    try:
        trend_summary = build_trend_summary_for_report(monthly_data_path, trend_end_file_name) if monthly_data_path else None
    except Exception:
        trend_summary = None

    latest_movement_text = "Monthly movement not available."
    if trend_summary:
        latest_movement_text = (
            f"Latest movement period: {trend_summary.get('previous_period')} to {trend_summary.get('latest_period')}.\n"
            f"- Cumulative import change: {fmt_money(trend_summary.get('import_change', 0))}\n"
            f"- Cumulative export change: {fmt_money(trend_summary.get('export_change', 0))}\n"
            f"- Cumulative trade deficit change: {fmt_money(trend_summary.get('deficit_change', 0))}\n"
            f"- Top rising import: {trend_summary.get('top_rising_import')} ({fmt_money(trend_summary.get('top_rising_import_change', 0))})\n"
            f"- Top falling import: {trend_summary.get('top_falling_import')} ({fmt_money(trend_summary.get('top_falling_import_change', 0))})\n"
            f"- Top rising export: {trend_summary.get('top_rising_export')} ({fmt_money(trend_summary.get('top_rising_export_change', 0))})\n"
            f"- Top falling export: {trend_summary.get('top_falling_export')} ({fmt_money(trend_summary.get('top_falling_export_change', 0))})\n"
            f"- Note: {trend_summary.get('movement_note')}"
        )

    monthly_trends = monthly_trend_table_text(n=20)
    import_movement = product_movement_text("5_Imports_By_Commodity", "Imports_Value", "import", movement_n)
    export_movement = product_movement_text("7_Exports_By_Commodity", "Exports_Value", "export", movement_n)

    context = f"""
TradePulse Nepal detailed processed context
Source: Department of Customs, Nepal
Dashboard file: {source_file_label}
Current period/column: {current_period_display}
Latest monthly file: {latest_month_label}
Monthly files loaded: {monthly_files_count}
Units: Rs. billion converted from source values in Rs. thousands
Context mode: {context_mode}. This is processed dashboard data, not raw Excel.

Core snapshot:
- Imports: {fmt_money(imports_total)}
- Exports: {fmt_money(exports_total)}
- Trade deficit: {fmt_money(deficit_total)}
- Total foreign trade: {fmt_money(total_trade)}
- Import-export ratio: {import_export_ratio:,.1f}x

Latest monthly movement summary:
{latest_movement_text}

Monthly trend table:
{monthly_trends}

Top import products by cumulative value:
{top_rows_text(imports, "Description", "Imports_Billion", top_n)}

Top export products by cumulative value:
{top_rows_text(exports, "Description", "Exports_Billion", top_n)}

Product movement from latest monthly files:
{import_movement}

{export_movement}

Top import partners:
{top_rows_text(countries, "Partner Countries", "Imports_Billion", top_n)}

Top export destinations:
{top_rows_text(countries, "Partner Countries", "Exports_Billion", top_n)}

Top customs routes by imports:
{top_rows_text(customs, "Customs", "Imports_Billion", min(top_n, 20))}

Sector summary:
{sector_rows_text(sector_summary, min(top_n, 20))}
""".strip()

    return context[:cap]

def generate_gemini_trade_answer(question, data_context, api_key, model_name="gemini-3.1-flash-lite", timeout_seconds=20):
    """Generate a Gemini answer with strict network timeout and graceful fallback."""
    if not api_key:
        return "Gemini API key is missing. Add GEMINI_API_KEY in Streamlit Secrets to use this tab."

    try:
        from google import genai
        from google.genai import types
    except Exception:
        return "The google-genai package is missing. Add `google-genai` to requirements.txt and redeploy."

    system_rules = """
You are TradePulse Nepal's AI trade analyst.
Answer only from the provided processed TradePulse dashboard context.
Do not invent numbers, dates, sources, products, sectors, countries, or forecasts.
If the context does not contain enough information, say the data is not available in the current dashboard.
Use Rs. billion where values are provided that way.
Do not give investment advice.
Use this format:
Direct answer:
What the data suggests:
Why it matters:
Caution:
Keep the answer under 240 words.
""".strip()

    prompt = f"""
{system_rules}

DATA CONTEXT:
{data_context}

USER QUESTION:
{question}
""".strip()

    try:
        # Set timeout inside the Google GenAI client itself. This is safer on Streamlit
        # Cloud than wrapping the request in a Python thread, because the old thread can
        # continue running after a timeout and make the app feel frozen.
        try:
            http_options = types.HttpOptions(timeout=timeout_seconds * 1000)
            client = genai.Client(api_key=api_key, http_options=http_options)
        except Exception:
            client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=520,
            ),
        )
        answer = getattr(response, "text", "")
        if answer:
            return answer.strip()
        return "Gemini returned an empty answer. Please try again later or use the stable Ask TradePulse tab."

    except Exception as exc:
        error_text = str(exc)
        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text or "quota" in error_text.lower():
            return "Gemini free quota is temporarily unavailable. Please try again later or use the stable Ask TradePulse tab."
        if "timeout" in error_text.lower() or "deadline" in error_text.lower() or "504" in error_text:
            return "Gemini took too long to respond. Please try again later or use the stable Ask TradePulse tab."
        if "NOT_FOUND" in error_text or "404" in error_text:
            return "The selected Gemini model is not available for this API key. Try another model option."
        return "Gemini could not generate an answer safely. Please try again later or use the stable Ask TradePulse tab."

# -----------------------------
# Prepare data
# -----------------------------

# Find the latest/current value column dynamically. Never hard-code a fiscal year.
possible_year_cols = []

for col in trade.columns:
    col_text = str(col).strip().lower()

    if col == "Trade.Indicators":
        continue
    if "change" in col_text:
        continue
    if col_text in ["sn", "s n", "s.n", "s.no", "s no"]:
        continue

    numeric_series = pd.to_numeric(trade[col], errors="coerce")
    if numeric_series.notna().sum() >= 3:
        possible_year_cols.append(col)

if not possible_year_cols:
    st.error("No usable fiscal-year value column was found in 1_Trade_Direction.")
    st.stop()


def trade_column_sort_key(column_name, fallback_index):
    text_value = str(column_name).strip()
    fy_match = re.search(r"(20\d{2})\s*/\s*(\d{2,4})", text_value)
    if fy_match:
        return 3, int(fy_match.group(1)), fallback_index
    if text_value.lower() == "current" or "current" in text_value.lower():
        return 4, 9999, fallback_index
    return 1, fallback_index, fallback_index

current_col = max(
    possible_year_cols,
    key=lambda col: trade_column_sort_key(col, possible_year_cols.index(col))
)
previous_cols = [col for col in possible_year_cols if col != current_col]
previous_col = previous_cols[-1] if previous_cols else None

# Normalize spacing only for presentation; keep the exact column name for lookups.
current_period_display = re.sub(r"\s+", " ", str(current_col)).strip()

imports_total_raw = get_trade_value(r"^Imports\s*\(", current_col)
exports_total_raw = get_trade_value(r"^Exports\s*\(", current_col)
deficit_total_raw = get_trade_value(r"Trade\s+Deficit", current_col)
total_trade_raw = get_trade_value(r"Total\s+Foreign\s+Trade", current_col)

imports_total = rs_thousand_to_billion(imports_total_raw)
exports_total = rs_thousand_to_billion(exports_total_raw)
deficit_total = rs_thousand_to_billion(deficit_total_raw)
total_trade = rs_thousand_to_billion(total_trade_raw)

import_export_ratio = imports_total_raw / exports_total_raw if exports_total_raw else 0


def percent_change(current_value, previous_value):
    if previous_value in (None, 0):
        return None
    return ((current_value - previous_value) / abs(previous_value)) * 100


def delta_text(delta_value, positive_word="vs prior comparable period"):
    if delta_value is None:
        return "Current release baseline"
    arrow = "↑" if delta_value > 0 else "↓" if delta_value < 0 else "→"
    sign = "+" if delta_value > 0 else ""
    return f"{arrow} {sign}{delta_value:,.1f}% {positive_word}"

if previous_col is not None:
    previous_imports_raw = get_trade_value(r"^Imports\s*\(", previous_col)
    previous_exports_raw = get_trade_value(r"^Exports\s*\(", previous_col)
    previous_deficit_raw = get_trade_value(r"Trade\s+Deficit", previous_col)
    previous_ratio_raw = get_trade_value(r"Imports/Exports\s+Ratio|Imports\s*/\s*Exports\s+Ratio", previous_col)

    imports_delta_pct = percent_change(imports_total_raw, previous_imports_raw)
    exports_delta_pct = percent_change(exports_total_raw, previous_exports_raw)
    deficit_delta_pct = percent_change(deficit_total_raw, previous_deficit_raw)
    ratio_delta_pct = percent_change(import_export_ratio, previous_ratio_raw if previous_ratio_raw else None)
else:
    imports_delta_pct = exports_delta_pct = deficit_delta_pct = ratio_delta_pct = None

imports = imports.copy()
exports = exports.copy()
countries = countries.copy()
customs = customs.copy()
import_partner = import_partner.copy()
export_partner = export_partner.copy()

imports["HSCode"] = clean_hscode(imports["HSCode"])
exports["HSCode"] = clean_hscode(exports["HSCode"])
import_partner["HSCode"] = clean_hscode(import_partner["HSCode"])
export_partner["HSCode"] = clean_hscode(export_partner["HSCode"])


imports["Imports_Billion"] = imports["Imports_Value"] / 1_000_000
imports["Revenue_Billion"] = imports["Imports_Revenue"] / 1_000_000
imports["Approx_Duty_Rate"] = (imports["Imports_Revenue"] / imports["Imports_Value"]) * 100
imports["Approx_Duty_Rate"] = imports["Approx_Duty_Rate"].replace([float("inf"), -float("inf")], 0).fillna(0)

exports["Exports_Billion"] = exports["Exports_Value"] / 1_000_000

countries["Imports_Billion"] = countries["Imports_Value"] / 1_000_000
countries["Exports_Billion"] = countries["Exports_Value"] / 1_000_000
countries["Trade_Balance_Billion"] = countries["Trade_Balance"] / 1_000_000

customs["Imports_Billion"] = customs["Imports_Value"] / 1_000_000
customs["Exports_Billion"] = customs["Exports_Value"] / 1_000_000

import_partner["Imports_Billion"] = import_partner["Imports_Value"] / 1_000_000
import_partner["Revenue_Billion"] = import_partner["Imports_Revenue"] / 1_000_000

export_partner["Exports_Billion"] = export_partner["Exports_Value"] / 1_000_000
# Remove total rows so dashboard shows real products/countries/routes
imports = remove_total_rows(imports, ["Description"])
exports = remove_total_rows(exports, ["Description"])
countries = remove_total_rows(countries, ["Partner Countries"])
customs = remove_total_rows(customs, ["Customs"])

import_partner = remove_total_rows(import_partner, ["Description", "Partner Countries"])
export_partner = remove_total_rows(export_partner, ["Description", "Partner Countries"])

top_imports = imports.sort_values("Imports_Value", ascending=False).head(10)
top_exports = exports.sort_values("Exports_Value", ascending=False).head(10)
top_import_countries = countries.sort_values("Imports_Value", ascending=False).head(10)
top_export_countries = countries.sort_values("Exports_Value", ascending=False).head(10)
top_deficit_countries = countries.sort_values("Trade_Balance").head(10)
top_surplus_countries = countries.sort_values("Trade_Balance", ascending=False).head(10)
top_customs = customs.sort_values("Imports_Value", ascending=False).head(10)

top_import_product = top_imports.iloc[0]["Description"]
top_import_value = top_imports.iloc[0]["Imports_Billion"]

top_export_product = top_exports.iloc[0]["Description"]
top_export_value = top_exports.iloc[0]["Exports_Billion"]

top_import_country = top_import_countries.iloc[0]["Partner Countries"]
top_export_country = top_export_countries.iloc[0]["Partner Countries"]

top_customs_office = top_customs.iloc[0]["Customs"]
top_customs_value = top_customs.iloc[0]["Imports_Billion"]
top5_route_share = top_customs["Import_Share"].head(5).sum()

sector_summary = build_sector_summary(imports, exports)

automated_insights = build_automated_insights(
    imports_total=imports_total,
    exports_total=exports_total,
    deficit_total=deficit_total,
    total_trade=total_trade,
    import_export_ratio=import_export_ratio,
    top_import_product=top_import_product,
    top_import_value=top_import_value,
    top_export_product=top_export_product,
    top_export_value=top_export_value,
    top_import_country=top_import_country,
    top_export_country=top_export_country,
    top_customs_office=top_customs_office,
    top_customs_value=top_customs_value,
    top5_route_share=top5_route_share,
    imports=imports,
    exports=exports,
    countries=countries,
    monthly_data_path=monthly_data_path
)

# -----------------------------
# Hero + executive status
# -----------------------------

hero_fy = active_fy_label or current_period_display.split("(")[0].strip()
hero_month = latest_month_label if latest_month_label != "Not available" else "Current release"

safe_hero_fy = html.escape(str(hero_fy))
safe_hero_month = html.escape(str(hero_month))
safe_period = html.escape(current_period_display)
safe_source = html.escape(str(source_type_label))
safe_file = html.escape(str(source_file_label))

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-grid">
            <div>
                <div class="brand-lockup">
                    <div class="brand-mark">TP</div>
                    <div>
                        <div class="brand-name">TradePulse Nepal</div>
                        <div class="brand-line">Nepal · Trade Intelligence System</div>
                    </div>
                </div>
                <div class="hero-kicker">Department of Customs data · transformed into signals</div>
                <div class="hero-title">See Nepal's trade <span>before it becomes a headline.</span></div>
                <div class="hero-subtitle">
                    Explore products, sectors, partner countries, customs routes and market movement from Nepal's official trade releases —
                    with decision-ready signals for business, research and policy.
                </div>
                <div class="hero-meta">
                    <span class="meta-chip live">{safe_hero_fy}</span>
                    <span class="meta-chip">{safe_hero_month}</span>
                    <span class="meta-chip">{monthly_files_count} release{'s' if monthly_files_count != 1 else ''} loaded</span>
                    <span class="meta-chip">Rs billion view</span>
                </div>
            </div>
            <div class="pulse-panel">
                <div class="pulse-head"><span>Market pulse</span><span>LIVE RELEASE</span></div>
                <div class="pulse-period">{safe_period}</div>
                <div class="pulse-row">
                    <span>Imports</span><strong>Rs {imports_total:,.2f}B</strong>
                    <em>{html.escape(delta_text(imports_delta_pct))}</em>
                </div>
                <div class="pulse-row">
                    <span>Exports</span><strong>Rs {exports_total:,.2f}B</strong>
                    <em>{html.escape(delta_text(exports_delta_pct))}</em>
                </div>
                <div class="pulse-row">
                    <span>Trade deficit</span><strong>Rs {deficit_total:,.2f}B</strong>
                    <em>{html.escape(delta_text(deficit_delta_pct))}</em>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# KPI cards
# -----------------------------

k1, k2, k3, k4 = st.columns(4)

with k1:
    kpi_card("Total Imports", f"Rs {imports_total:,.2f}B", delta_text(imports_delta_pct))

with k2:
    kpi_card("Total Exports", f"Rs {exports_total:,.2f}B", delta_text(exports_delta_pct))

with k3:
    kpi_card("Trade Deficit", f"Rs {deficit_total:,.2f}B", delta_text(deficit_delta_pct))

with k4:
    kpi_card("Import / Export Ratio", f"{import_export_ratio:,.2f}x", delta_text(ratio_delta_pct))

st.markdown(
    f"""
    <div class="status-shell">
        <div class="status-head">
            <div class="status-title">Data release status</div>
            <div class="status-trust">● official-source workbook loaded</div>
        </div>
        <div class="status-grid">
            <div class="status-card">
                <div class="status-label">Dashboard source</div>
                <div class="status-value">{safe_source}</div>
            </div>
            <div class="status-card">
                <div class="status-label">Current period</div>
                <div class="status-value">{safe_period}</div>
            </div>
            <div class="status-card">
                <div class="status-label">Latest release</div>
                <div class="status-value">{safe_hero_month}</div>
            </div>
            <div class="status-card">
                <div class="status-label">FY releases loaded</div>
                <div class="status-value">{monthly_files_count}</div>
            </div>
        </div>
        <div class="status-note">
            Using <b>{safe_file}</b>. Source monetary values are Rs. thousands and are converted to Rs. billion for presentation.
            Insights and AI features use the processed Customs data available in this app.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Tabs
# -----------------------------

overview_tab, product_tab, sector_tab, detail_tab, opportunity_tab, country_tab, route_tab, trend_tab, about_tab, insight_tab, ask_tab, gemini_tab = st.tabs(
    [
        "Overview",
        "Products",
        "Sectors",
        "Product Detail",
        "Opportunity Finder",
        "Countries",
        "Customs Routes",
        "Trends",
        "About / Methodology",
        "Insights",
        "Ask TradePulse",
        "Gemini AI Analyst"
    ]
)

# -----------------------------
# Overview tab
# -----------------------------

with overview_tab:
    st.subheader("Market Snapshot")

    c1, c2 = st.columns([1.2, 1])

    with c1:
        summary_df = pd.DataFrame({
            "Indicator": ["Imports", "Exports", "Trade Deficit", "Total Foreign Trade"],
            "Rs Billion": [imports_total, exports_total, deficit_total, total_trade]
        })

        fig = px.bar(
            summary_df,
            x="Indicator",
            y="Rs Billion",
            text="Rs Billion",
            title="Foreign Trade Snapshot"
        )

        fig.update_traces(
            texttemplate="%{text:,.1f}",
            textposition="outside"
        )

        fig.update_layout(
            height=430,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            title_font_size=20,
            font=dict(color="#102A43"),
            yaxis=dict(gridcolor="rgba(16,42,67,0.08)")
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown(
            f"""
            <div class="section-card">
                <h3>What this means</h3>
                <p>Nepal's imports are around <b>{import_export_ratio:,.1f} times</b> larger than exports.</p>
                <p>The trade deficit of <b>Rs {deficit_total:,.2f} billion</b> is the main pressure point in the trade account.</p>
                <p>The dashboard helps identify <b>which products, countries, and customs routes</b> explain this movement.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("Top Signals")

    s1, s2, s3 = st.columns(3)

    with s1:
        st.info(f"Top import product: **{top_import_product}** — Rs {top_import_value:,.2f}B")

    with s2:
        st.success(f"Top export product: **{top_export_product}** — Rs {top_export_value:,.2f}B")

    with s3:
        st.warning(f"Top import country: **{top_import_country}**")

# -----------------------------
# Product tab
# -----------------------------

with product_tab:
    st.subheader("Product Movement")

    p1, p2 = st.columns(2)

    with p1:
        fig1 = horizontal_bar(
            top_imports,
            "Imports_Billion",
            "Description",
            "Top 10 Imported Products",
            "Rs Billion",
            "Product"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with p2:
        fig2 = horizontal_bar(
            top_exports,
            "Exports_Billion",
            "Description",
            "Top 10 Exported Products",
            "Rs Billion",
            "Product"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Product Search")

    search_text = st.text_input(
        "Search commodity name or HS code",
        placeholder="Example: gold, petroleum, rice, 2710..."
    )

    if search_text:
        mask = (
            imports["Description"].astype(str).str.contains(search_text, case=False, na=False)
            | imports["HSCode"].astype(str).str.contains(search_text, case=False, na=False)
        )
        result = imports.loc[mask].sort_values("Imports_Value", ascending=False).head(30)
    else:
        result = top_imports

    st.dataframe(
        result[[
            "HSCode",
            "Description",
            "Unit",
            "Quantity",
            "Imports_Billion",
            "Revenue_Billion",
            "Approx_Duty_Rate"
        ]],
        use_container_width=True
    )

    st.download_button(
        "Download product search / top imports as CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="product_search_results.csv",
        mime="text/csv"
    )

# -----------------------------
# Opportunity Finder tab
# -----------------------------

with opportunity_tab:
    st.subheader("Opportunity Finder")

    st.markdown(
        """
        This section ranks imported products by market opportunity using customs data.
        The score is based on import value, customs revenue, and approximate duty signal.
        """
    )

    opportunity_df = imports.copy()

    opportunity_df = opportunity_df[
        opportunity_df["Imports_Billion"].notna() &
        (opportunity_df["Imports_Billion"] > 0)
    ].copy()

    opportunity_df["Market_Size_Score"] = opportunity_df["Imports_Billion"].rank(pct=True) * 100
    opportunity_df["Revenue_Signal_Score"] = opportunity_df["Revenue_Billion"].rank(pct=True) * 100
    opportunity_df["Duty_Signal_Score"] = opportunity_df["Approx_Duty_Rate"].rank(pct=True) * 100

    opportunity_df["Opportunity_Score"] = (
        opportunity_df["Market_Size_Score"] * 0.60 +
        opportunity_df["Revenue_Signal_Score"] * 0.25 +
        opportunity_df["Duty_Signal_Score"] * 0.15
    )

    high_import_threshold = opportunity_df["Imports_Billion"].quantile(0.90)

    def classify_opportunity(row):
        if row["Imports_Billion"] >= high_import_threshold and row["Approx_Duty_Rate"] >= 5:
            return "High import + duty signal"
        elif row["Imports_Billion"] >= high_import_threshold:
            return "Large import market"
        elif row["Opportunity_Score"] >= 85:
            return "Strong watchlist"
        elif row["Opportunity_Score"] >= 70:
            return "Medium watchlist"
        else:
            return "Low priority"

    opportunity_df["Opportunity_Type"] = opportunity_df.apply(classify_opportunity, axis=1)

    f1, f2, f3 = st.columns(3)

    with f1:
        min_import_value = st.number_input(
            "Minimum import value, Rs billion",
            min_value=0.0,
            value=1.0,
            step=0.5
        )

    with f2:
        top_n = st.slider(
            "Number of products to show",
            min_value=10,
            max_value=100,
            value=25,
            step=5
        )

    with f3:
        opportunity_search = st.text_input(
            "Search product / HS code",
            placeholder="gold, petroleum, rice, 2710...",
            key="opportunity_search"
        )

    filtered_opportunity = opportunity_df[
        opportunity_df["Imports_Billion"] >= min_import_value
    ].copy()

    if opportunity_search:
        filtered_opportunity = filtered_opportunity[
            filtered_opportunity["Description"].astype(str).str.contains(opportunity_search, case=False, na=False)
            | filtered_opportunity["HSCode"].astype(str).str.contains(opportunity_search, case=False, na=False)
        ]

    filtered_opportunity = filtered_opportunity.sort_values(
        "Opportunity_Score",
        ascending=False
    ).head(top_n)

    o1, o2, o3, o4 = st.columns(4)

    with o1:
        kpi_card("Products Screened", f"{len(opportunity_df):,}", "Imported products with positive value")

    with o2:
        kpi_card("Shown Products", f"{len(filtered_opportunity):,}", "After filters")

    with o3:
        if len(filtered_opportunity) > 0:
            kpi_card("Top Score", f"{filtered_opportunity['Opportunity_Score'].max():,.1f}", "Out of 100")
        else:
            kpi_card("Top Score", "0", "No result")

    with o4:
        if len(filtered_opportunity) > 0:
            kpi_card("Largest Import", f"Rs {filtered_opportunity['Imports_Billion'].max():,.2f}B", "Within filtered list")
        else:
            kpi_card("Largest Import", "Rs 0B", "No result")

    st.markdown("### Top Opportunity Products")

    display_cols = [
        "HSCode",
        "Description",
        "Unit",
        "Imports_Billion",
        "Revenue_Billion",
        "Approx_Duty_Rate",
        "Market_Size_Score",
        "Revenue_Signal_Score",
        "Duty_Signal_Score",
        "Opportunity_Score",
        "Opportunity_Type"
    ]

    if len(filtered_opportunity) == 0:
        st.warning("No products found. Try reducing the minimum import value or clearing the search box.")
    else:
        st.dataframe(
            filtered_opportunity[display_cols].style.format({
                "Imports_Billion": "{:,.2f}",
                "Revenue_Billion": "{:,.2f}",
                "Approx_Duty_Rate": "{:,.2f}",
                "Market_Size_Score": "{:,.1f}",
                "Revenue_Signal_Score": "{:,.1f}",
                "Duty_Signal_Score": "{:,.1f}",
                "Opportunity_Score": "{:,.1f}"
            }),
            use_container_width=True
        )

        st.markdown("### Opportunity Map")

        fig_opportunity = px.scatter(
            filtered_opportunity,
            x="Imports_Billion",
            y="Approx_Duty_Rate",
            size="Opportunity_Score",
            hover_name="Description",
            hover_data={
                "HSCode": True,
                "Imports_Billion": ":,.2f",
                "Revenue_Billion": ":,.2f",
                "Approx_Duty_Rate": ":,.2f",
                "Opportunity_Score": ":,.1f"
            },
            title="Import Opportunity Map",
            labels={
                "Imports_Billion": "Import Value, Rs Billion",
                "Approx_Duty_Rate": "Approx. Duty / Revenue Rate, %"
            }
        )

        fig_opportunity.update_layout(
            height=540,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            title_font_size=20,
            font=dict(color="#102A43"),
            xaxis=dict(gridcolor="rgba(16,42,67,0.08)"),
            yaxis=dict(gridcolor="rgba(16,42,67,0.08)")
        )

        st.plotly_chart(fig_opportunity, use_container_width=True)

        st.markdown(
            """
            <div class="opportunity-card">
                <b>How to read this:</b><br>
                Higher import value means the product already has a large market in Nepal.<br>
                Higher revenue or duty signal may indicate policy sensitivity, tariff protection, or strong taxable import flow.<br>
                Higher opportunity score means the product deserves deeper business, production, import-substitution, or policy analysis.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.download_button(
            "Download opportunity products as CSV",
            data=filtered_opportunity[display_cols].to_csv(index=False).encode("utf-8"),
            file_name="opportunity_finder_products.csv",
            mime="text/csv"
        )


# -----------------------------
# Sector Dashboard tab
# -----------------------------

with sector_tab:
    st.subheader("Sector Dashboard")

    st.markdown(
        """
        This section groups HS-code products into broad sectors so users can quickly see
        where Nepal's import pressure, export strength, and sector-level trade gaps are concentrated.
        """
    )

    if sector_summary.empty:
        st.warning("Sector summary could not be created from the current customs file.")
    else:
        sector_k1, sector_k2, sector_k3, sector_k4 = st.columns(4)

        top_import_sector = sector_summary.sort_values("Imports_Billion", ascending=False).iloc[0]
        top_export_sector = sector_summary.sort_values("Exports_Billion", ascending=False).iloc[0]
        top_gap_sector = sector_summary.sort_values("Sector_Gap_Billion", ascending=False).iloc[0]

        with sector_k1:
            kpi_card("Top Import Sector", short_text(top_import_sector["Sector"], 32), f"Rs {top_import_sector['Imports_Billion']:,.2f}B")
        with sector_k2:
            kpi_card("Top Export Sector", short_text(top_export_sector["Sector"], 32), f"Rs {top_export_sector['Exports_Billion']:,.2f}B")
        with sector_k3:
            kpi_card("Largest Sector Gap", short_text(top_gap_sector["Sector"], 32), f"Rs {top_gap_sector['Sector_Gap_Billion']:,.2f}B")
        with sector_k4:
            kpi_card("Sectors Covered", f"{len(sector_summary):,}", "HS chapters grouped into sectors")

        st.markdown("### Sector Trade Structure")

        sector_chart_df = sector_summary.sort_values("Imports_Billion", ascending=True)

        fig_sector = px.bar(
            sector_chart_df,
            x=["Imports_Billion", "Exports_Billion"],
            y="Sector",
            orientation="h",
            title="Sector-wise Imports and Exports",
            labels={
                "value": "Rs Billion",
                "Sector": "Sector",
                "variable": "Trade flow"
            }
        )

        fig_sector.update_layout(
            height=620,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            title_font_size=20,
            font=dict(color="#102A43"),
            xaxis=dict(gridcolor="rgba(16,42,67,0.08)")
        )

        st.plotly_chart(fig_sector, use_container_width=True)

        st.markdown("### Sector Table")

        display_sector = sector_summary.copy()
        display_sector["Signal"] = display_sector.apply(sector_signal, axis=1)

        st.dataframe(
            display_sector[[
                "Sector",
                "Imports_Billion",
                "Exports_Billion",
                "Sector_Gap_Billion",
                "Import_Share",
                "Export_Share",
                "Signal"
            ]].style.format({
                "Imports_Billion": "{:,.2f}",
                "Exports_Billion": "{:,.2f}",
                "Sector_Gap_Billion": "{:,.2f}",
                "Import_Share": "{:,.1f}%",
                "Export_Share": "{:,.1f}%"
            }),
            use_container_width=True
        )

        st.download_button(
            "Download sector summary as CSV",
            data=display_sector.to_csv(index=False).encode("utf-8"),
            file_name="tradepulse_sector_summary.csv",
            mime="text/csv"
        )

        st.markdown("### Explore a Sector")

        selected_sector = st.selectbox(
            "Choose sector",
            sector_summary["Sector"].tolist(),
            key="sector_selector"
        )

        selected_sector_row = sector_summary[sector_summary["Sector"] == selected_sector].iloc[0]

        st.info(sector_signal(selected_sector_row))

        sp1, sp2 = st.columns(2)

        sector_top_imports, sector_top_exports = get_sector_top_products(
            imports,
            exports,
            selected_sector,
            top_n=10
        )

        with sp1:
            st.markdown("#### Top import products in this sector")
            if len(sector_top_imports) == 0:
                st.caption("No import products found for this sector.")
            else:
                st.dataframe(
                    sector_top_imports[["HSCode", "Description", "Imports_Billion"]].style.format({
                        "Imports_Billion": "{:,.2f}"
                    }),
                    use_container_width=True
                )

        with sp2:
            st.markdown("#### Top export products in this sector")
            if len(sector_top_exports) == 0:
                st.caption("No export products found for this sector.")
            else:
                st.dataframe(
                    sector_top_exports[["HSCode", "Description", "Exports_Billion"]].style.format({
                        "Exports_Billion": "{:,.2f}"
                    }),
                    use_container_width=True
                )


# -----------------------------
# Product Detail tab
# -----------------------------

with detail_tab:
    st.subheader("Product Detail Page")

    st.markdown(
        """
        Search a product or HS code to see its current import/export value, sector,
        partner countries, and monthly movement where available.
        """
    )

    product_mode = st.radio(
        "Product direction",
        ["Import product", "Export product"],
        horizontal=True,
        key="product_detail_mode"
    )

    if product_mode == "Import product":
        product_source = imports.copy()
        value_col_for_sort = "Imports_Billion"
        direction = "import"
    else:
        product_source = exports.copy()
        value_col_for_sort = "Exports_Billion"
        direction = "export"

    product_source["Search_Label"] = (
        product_source["HSCode"].astype(str)
        + " · "
        + product_source["Description"].astype(str)
    )

    product_search = st.text_input(
        "Search product name or HS code",
        placeholder="Example: diesel, gold, rice, 2710, tea...",
        key="product_detail_search"
    )

    if product_search:
        product_filtered = product_source[
            product_source["Description"].astype(str).str.contains(product_search, case=False, na=False)
            | product_source["HSCode"].astype(str).str.contains(product_search, case=False, na=False)
        ].copy()
    else:
        product_filtered = product_source.sort_values(value_col_for_sort, ascending=False).head(100).copy()

    product_filtered = product_filtered.sort_values(value_col_for_sort, ascending=False)

    if len(product_filtered) == 0:
        st.warning("No product found. Try another name or HS code.")
    else:
        selected_label = st.selectbox(
            "Select product",
            product_filtered["Search_Label"].tolist(),
            key="product_detail_selector"
        )

        selected_product = product_filtered[
            product_filtered["Search_Label"] == selected_label
        ].iloc[0]

        detail = product_detail_summary(selected_product, direction=direction)

        d1, d2, d3, d4 = st.columns(4)

        with d1:
            kpi_card(detail["value_label"], f"Rs {float(detail['value']):,.2f}B", "Current dashboard period")
        with d2:
            kpi_card("Sector", short_text(detail["sector"], 34), "Based on HS chapter")
        with d3:
            kpi_card("HS Code", str(detail["hs"]), "Commodity classification")
        with d4:
            kpi_card("Quantity", f"{detail['quantity']}", f"Unit: {detail['unit']}")

        st.markdown("### Product Analyst Note")

        st.markdown(
            f"""
            <div class="section-card">
                <h3>{clean_display(detail['description'])}</h3>
                <p><b>Sector:</b> {clean_display(detail['sector'])}</p>
                <p><b>Data signal:</b> {clean_display(detail['note'])}</p>
                <p><b>Caution:</b> This is a screening view from customs data only. It does not include domestic production, prices, margins, or firm-level data.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Partner Countries")

        if direction == "import":
            partner_table = product_partner_table(
                import_partner,
                detail["hs"],
                "Imports_Value",
                "Imports_Billion",
                top_n=10
            )
            partner_value_col = "Imports_Billion"
            partner_title = "Top import partner countries"
        else:
            partner_table = product_partner_table(
                export_partner,
                detail["hs"],
                "Exports_Value",
                "Exports_Billion",
                top_n=10
            )
            partner_value_col = "Exports_Billion"
            partner_title = "Top export destination countries"

        if len(partner_table) == 0:
            st.caption("Partner-country detail was not found for this product in the current file.")
        else:
            st.markdown(f"#### {partner_title}")

            partner_table = partner_table.copy()
            if partner_value_col in partner_table.columns:
                partner_table[partner_value_col] = pd.to_numeric(partner_table[partner_value_col], errors="coerce")

            fig_partner = px.bar(
                partner_table.sort_values(partner_value_col, ascending=True),
                x=partner_value_col,
                y="Partner Countries",
                orientation="h",
                title=partner_title,
                labels={
                    partner_value_col: "Rs Billion",
                    "Partner Countries": "Partner country"
                }
            )

            fig_partner.update_layout(
                height=460,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                title_font_size=20,
                font=dict(color="#102A43"),
                xaxis=dict(gridcolor="rgba(16,42,67,0.08)")
            )

            st.plotly_chart(fig_partner, use_container_width=True)

            st.dataframe(
                partner_table.style.format({partner_value_col: "{:,.2f}"}),
                use_container_width=True
            )

        st.markdown("### Monthly Movement")

        movement_table = product_monthly_movement(
            monthly_data_path,
            detail["hs"],
            direction=direction,
            end_file_name=selected_trend_end_name
        )

        if len(movement_table) == 0:
            st.caption("Monthly movement was not available for this product from the monthly_data folder.")
        else:
            movement_row = movement_table.iloc[0]

            m1, m2, m3 = st.columns(3)
            with m1:
                kpi_card("Latest Month", f"Rs {movement_row['Latest_Month_Billion']:,.2f}B", str(movement_row["Latest_Period"]))
            with m2:
                kpi_card("Previous Month", f"Rs {movement_row['Previous_Month_Billion']:,.2f}B", str(movement_row["Previous_Period"]))
            with m3:
                kpi_card("Monthly Change", f"Rs {movement_row['Change_Billion']:,.2f}B", "Latest minus previous")

            st.dataframe(
                movement_table[[
                    "HSCode",
                    "Description",
                    "Latest_Cumulative_Billion",
                    "Previous_Month_Billion",
                    "Latest_Month_Billion",
                    "Change_Billion",
                    "Growth_Percent"
                ]].style.format({
                    "Latest_Cumulative_Billion": "{:,.2f}",
                    "Previous_Month_Billion": "{:,.2f}",
                    "Latest_Month_Billion": "{:,.2f}",
                    "Change_Billion": "{:,.2f}",
                    "Growth_Percent": "{:,.1f}%"
                }),
                use_container_width=True
            )

        st.markdown("### Route Context")

        st.caption(
            "The current product detail view uses overall customs-route data because the loaded workbook does not provide product-by-customs-route detail in the sheets used by this app."
        )

        st.dataframe(
            top_customs[["Customs", "Imports_Billion", "Exports_Billion", "Import_Share"]].style.format({
                "Imports_Billion": "{:,.2f}",
                "Exports_Billion": "{:,.2f}",
                "Import_Share": "{:,.1f}%"
            }),
            use_container_width=True
        )

        product_detail_export = pd.DataFrame([{
            "HSCode": detail["hs"],
            "Description": detail["description"],
            "Direction": direction,
            "Sector": detail["sector"],
            "Value_Billion": detail["value"],
            "Unit": detail["unit"],
            "Quantity": detail["quantity"]
        }])

        st.download_button(
            "Download product detail as CSV",
            data=product_detail_export.to_csv(index=False).encode("utf-8"),
            file_name="tradepulse_product_detail.csv",
            mime="text/csv"
        )


# -----------------------------
# Country tab
# -----------------------------

with country_tab:
    st.subheader("Country Intelligence")

    country_list = (
        countries["Partner Countries"]
        .dropna()
        .sort_values()
        .unique()
    )

    country_list_as_list = list(country_list)

    selected_country = st.selectbox(
        "Select a partner country",
        country_list,
        index=country_list_as_list.index("India") if "India" in country_list_as_list else 0
    )

    selected_country_row = countries[
        countries["Partner Countries"] == selected_country
    ].iloc[0]

    selected_imports = selected_country_row["Imports_Billion"]
    selected_exports = selected_country_row["Exports_Billion"]
    selected_balance = selected_country_row["Trade_Balance_Billion"]

    selected_import_share = (selected_imports / imports_total) * 100 if imports_total else 0
    selected_export_share = (selected_exports / exports_total) * 100 if exports_total else 0

    if selected_balance < 0:
        balance_status = "Trade Deficit"
        balance_note = f"Nepal imports more from {selected_country} than it exports to that country."
    else:
        balance_status = "Trade Surplus"
        balance_note = f"Nepal exports more to {selected_country} than it imports from that country."

    if selected_import_share >= 30:
        dependency_level = "High dependency"
    elif selected_import_share >= 10:
        dependency_level = "Medium dependency"
    else:
        dependency_level = "Low dependency"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("Imports From Country", f"Rs {selected_imports:,.2f}B", f"{selected_import_share:,.1f}% of total imports")

    with c2:
        kpi_card("Exports To Country", f"Rs {selected_exports:,.2f}B", f"{selected_export_share:,.1f}% of total exports")

    with c3:
        kpi_card(balance_status, f"Rs {abs(selected_balance):,.2f}B", "Absolute trade balance")

    with c4:
        kpi_card("Dependency Level", dependency_level, "Based on import share")

    st.markdown("### Country Profile")

    profile_col1, profile_col2 = st.columns([1.1, 1])

    with profile_col1:
        country_mix = pd.DataFrame({
            "Flow": ["Imports", "Exports"],
            "Rs Billion": [selected_imports, selected_exports]
        })

        fig_country_mix = px.bar(
            country_mix,
            x="Flow",
            y="Rs Billion",
            text="Rs Billion",
            title=f"Nepal's Trade With {selected_country}"
        )

        fig_country_mix.update_traces(
            texttemplate="%{text:,.2f}",
            textposition="outside"
        )

        fig_country_mix.update_layout(
            height=420,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            title_font_size=20,
            font=dict(color="#102A43"),
            yaxis=dict(gridcolor="rgba(16,42,67,0.08)")
        )

        st.plotly_chart(fig_country_mix, use_container_width=True)

    with profile_col2:
        st.markdown(
            f"""
            <div class="section-card">
                <h3>{selected_country} — Trade Interpretation</h3>
                <p><b>{balance_status}:</b> {balance_note}</p>
                <p><b>Import dependence:</b> {selected_country} accounts for around <b>{selected_import_share:,.1f}%</b> of Nepal's total imports.</p>
                <p><b>Export dependence:</b> {selected_country} accounts for around <b>{selected_export_share:,.1f}%</b> of Nepal's total exports.</p>
                <p><b>Risk signal:</b> {dependency_level}.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Top Products Linked With Selected Country")

    selected_import_products = import_partner[
        import_partner["Partner Countries"] == selected_country
    ].sort_values("Imports_Value", ascending=False).head(15)

    selected_export_products = export_partner[
        export_partner["Partner Countries"] == selected_country
    ].sort_values("Exports_Value", ascending=False).head(15)

    p1, p2 = st.columns(2)

    with p1:
        st.markdown(f"#### Top Imports From {selected_country}")

        if len(selected_import_products) == 0:
            st.warning("No import product data found for this country.")
        else:
            fig_import_country_products = horizontal_bar(
                selected_import_products,
                "Imports_Billion",
                "Description",
                f"Top Imports From {selected_country}",
                "Rs Billion",
                "Product",
                height=520
            )

            st.plotly_chart(fig_import_country_products, use_container_width=True)

            st.dataframe(
                selected_import_products[[
                    "HSCode",
                    "Description",
                    "Unit",
                    "Quantity",
                    "Imports_Billion",
                    "Revenue_Billion"
                ]],
                use_container_width=True
            )

    with p2:
        st.markdown(f"#### Top Exports To {selected_country}")

        if len(selected_export_products) == 0:
            st.warning("No export product data found for this country.")
        else:
            fig_export_country_products = horizontal_bar(
                selected_export_products,
                "Exports_Billion",
                "Description",
                f"Top Exports To {selected_country}",
                "Rs Billion",
                "Product",
                height=520
            )

            st.plotly_chart(fig_export_country_products, use_container_width=True)

            st.dataframe(
                selected_export_products[[
                    "HSCode",
                    "Description",
                    "Unit",
                    "Quantity",
                    "Exports_Billion"
                ]],
                use_container_width=True
            )

    st.markdown("### Country-Level Opportunity and Risk Signals")

    signal_col1, signal_col2 = st.columns(2)

    with signal_col1:
        if len(selected_import_products) > 0:
            top_country_import_product = selected_import_products.iloc[0]["Description"]
            top_country_import_value = selected_import_products.iloc[0]["Imports_Billion"]

            st.markdown(
                f"""
                <div class="opportunity-card">
                    <h3>Business Signal</h3>
                    <p>The largest import from <b>{selected_country}</b> is <b>{top_country_import_product}</b>,
                    worth around <b>Rs {top_country_import_value:,.2f} billion</b>.</p>
                    <p>This product may deserve deeper study for sourcing, distribution, import financing,
                    or import-substitution potential.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("No business signal available because import product data was not found.")

    with signal_col2:
        st.markdown(
            f"""
            <div class="risk-card">
                <h3>Risk Signal</h3>
                <p>{selected_country} represents around <b>{selected_import_share:,.1f}%</b> of Nepal's import bill.</p>
                <p>If this share is high, Nepal may face supplier-country dependence, price shock exposure,
                exchange-rate pressure, or logistics concentration risk.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### Overall Country Rankings")

    c1, c2 = st.columns(2)

    with c1:
        fig3 = horizontal_bar(
            top_import_countries,
            "Imports_Billion",
            "Partner Countries",
            "Top Import Countries",
            "Rs Billion",
            "Country"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        fig4 = horizontal_bar(
            top_export_countries,
            "Exports_Billion",
            "Partner Countries",
            "Top Export Countries",
            "Rs Billion",
            "Country"
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Trade Balance Pressure")

    b1, b2 = st.columns(2)

    with b1:
        st.markdown("#### Biggest Deficit Countries")
        st.dataframe(
            top_deficit_countries[[
                "Partner Countries",
                "Imports_Billion",
                "Exports_Billion",
                "Trade_Balance_Billion"
            ]],
            use_container_width=True
        )

    with b2:
        st.markdown("#### Surplus Countries")
        st.dataframe(
            top_surplus_countries[[
                "Partner Countries",
                "Imports_Billion",
                "Exports_Billion",
                "Trade_Balance_Billion"
            ]],
            use_container_width=True
        )

# -----------------------------
# Customs route tab
# -----------------------------

with route_tab:
    st.subheader("Customs Route Intelligence")

    r1, r2 = st.columns([1.3, 1])

    with r1:
        fig5 = horizontal_bar(
            top_customs,
            "Imports_Billion",
            "Customs",
            "Top Customs Offices by Import Value",
            "Rs Billion",
            "Customs Office"
        )
        st.plotly_chart(fig5, use_container_width=True)

    with r2:
        st.markdown(
            f"""
            <div class="section-card">
                <h3>Route Concentration</h3>
                <p>The top 5 customs offices handle around <b>{top5_route_share:,.1f}%</b> of import value.</p>
                <p>The largest import route is <b>{top_customs_office}</b>, handling around <b>Rs {top_customs_value:,.2f} billion</b>.</p>
                <p>This can be used as a logistics risk and infrastructure pressure indicator.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.dataframe(
        customs.sort_values("Imports_Value", ascending=False)[[
            "Customs",
            "Imports_Billion",
            "Import_Share",
            "Exports_Billion",
            "Export_Share"
        ]],
        use_container_width=True
    )


# -----------------------------
# Trends tab
# -----------------------------

with trend_tab:
    st.subheader("Multi-Month Trade Trends")

    st.markdown(
        """
        <div class="insight-card">
            <b>How this works:</b> This tab automatically reads all monthly customs Excel files
            stored inside the <b>monthly_data</b> folder in this app. No upload is needed.
            For correct ordering, name files like <b>01_Shravan.xlsx</b>, <b>02_Bhadra.xlsx</b>,
            ... <b>11_Jestha.xlsx</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

    if not monthly_data_path.exists():
        st.warning("No active monthly-data folder was found. Add monthly_data/<FY>/ and place monthly Customs workbooks inside it.")
    else:
        trend_files = get_trend_files_for_period(monthly_data_path, selected_trend_end_name)

        if len(trend_files) == 0:
            st.info("No monthly Customs files are available in the active fiscal-year folder yet.")
        elif len(trend_files) < 2:
            try:
                baseline = extract_trade_snapshot_from_file(trend_files[0], trend_files[0].name)
                st.markdown(
                    f"""
                    <div class="launch-state">
                        <div class="launch-tag">Fiscal year baseline established</div>
                        <h3>{html.escape(str(baseline['Period']))} is the first release in this fiscal year.</h3>
                        <p>Month-on-month movement needs a second cumulative Customs release. TradePulse will activate the full trend engine automatically when the next workbook is added to <b>{html.escape(str(monthly_data_path.name))}</b>. The current release is already available across Overview, Products, Sectors, Countries, Customs Routes, Insights and AI tools.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    kpi_card("Baseline Imports", f"Rs {baseline['Imports_Billion']:,.2f}B", "First fiscal-month release")
                with b2:
                    kpi_card("Baseline Exports", f"Rs {baseline['Exports_Billion']:,.2f}B", "First fiscal-month release")
                with b3:
                    kpi_card("Baseline Deficit", f"Rs {baseline['Trade_Deficit_Billion']:,.2f}B", "First fiscal-month release")
                with b4:
                    kpi_card("Trend Readiness", "1 / 2", "Add the next month to unlock movement")
            except Exception as e:
                st.warning(f"The first monthly release exists but could not be processed for the trend baseline: {e}")
        else:
            trend_rows = []

            for file_path in trend_files:
                try:
                    row = extract_trade_snapshot_from_file(file_path, file_path.name)
                    trend_rows.append(row)
                except Exception as e:
                    st.warning(f"Could not process {file_path.name}: {e}")

            if len(trend_rows) < 2:
                st.warning("Could not process enough files for trend comparison.")
            else:
                trend_df = pd.DataFrame(trend_rows)
                trend_df["Order"] = range(1, len(trend_df) + 1)

                # Department of Customs monthly files are often cumulative.
                # Monthly movement is estimated by differencing consecutive cumulative files.
                trend_df["Monthly_Imports_Billion"] = trend_df["Imports_Billion"].diff()
                trend_df["Monthly_Exports_Billion"] = trend_df["Exports_Billion"].diff()
                trend_df["Monthly_Deficit_Billion"] = trend_df["Trade_Deficit_Billion"].diff()
                trend_df["Monthly_Total_Trade_Billion"] = trend_df["Total_Trade_Billion"].diff()

                trend_df.loc[trend_df.index[0], "Monthly_Imports_Billion"] = trend_df.loc[trend_df.index[0], "Imports_Billion"]
                trend_df.loc[trend_df.index[0], "Monthly_Exports_Billion"] = trend_df.loc[trend_df.index[0], "Exports_Billion"]
                trend_df.loc[trend_df.index[0], "Monthly_Deficit_Billion"] = trend_df.loc[trend_df.index[0], "Trade_Deficit_Billion"]
                trend_df.loc[trend_df.index[0], "Monthly_Total_Trade_Billion"] = trend_df.loc[trend_df.index[0], "Total_Trade_Billion"]

                latest = trend_df.iloc[-1]
                previous = trend_df.iloc[-2]

                import_change = latest["Imports_Billion"] - previous["Imports_Billion"]
                export_change = latest["Exports_Billion"] - previous["Exports_Billion"]
                deficit_change = latest["Trade_Deficit_Billion"] - previous["Trade_Deficit_Billion"]

                t1, t2, t3, t4 = st.columns(4)

                with t1:
                    kpi_card(
                        "Latest Imports",
                        f"Rs {latest['Imports_Billion']:,.2f}B",
                        f"Previous change: Rs {import_change:,.2f}B"
                    )

                with t2:
                    kpi_card(
                        "Latest Exports",
                        f"Rs {latest['Exports_Billion']:,.2f}B",
                        f"Previous change: Rs {export_change:,.2f}B"
                    )

                with t3:
                    kpi_card(
                        "Latest Deficit",
                        f"Rs {latest['Trade_Deficit_Billion']:,.2f}B",
                        f"Previous change: Rs {deficit_change:,.2f}B"
                    )

                with t4:
                    kpi_card(
                        "Files Used",
                        f"{len(trend_df)}",
                        "Monthly customs files"
                    )

                st.markdown("### Cumulative Trade Trend")

                cumulative_long = trend_df.melt(
                    id_vars=["Order", "Period"],
                    value_vars=[
                        "Imports_Billion",
                        "Exports_Billion",
                        "Trade_Deficit_Billion",
                        "Total_Trade_Billion"
                    ],
                    var_name="Indicator",
                    value_name="Rs_Billion"
                )

                fig_cumulative = px.line(
                    cumulative_long,
                    x="Order",
                    y="Rs_Billion",
                    color="Indicator",
                    markers=True,
                    title="Cumulative Trade Movement",
                    labels={
                        "Order": "Month Order",
                        "Rs_Billion": "Rs Billion",
                        "Indicator": "Indicator"
                    },
                    hover_data=["Period"]
                )

                fig_cumulative.update_layout(
                    height=520,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    title_font_size=22,
                    font=dict(color="#0F172A"),
                    xaxis=dict(gridcolor="rgba(15,23,42,0.08)"),
                    yaxis=dict(gridcolor="rgba(15,23,42,0.08)")
                )

                st.plotly_chart(fig_cumulative, use_container_width=True)

                st.markdown("### Estimated Monthly Movement")

                monthly_long = trend_df.melt(
                    id_vars=["Order", "Period"],
                    value_vars=[
                        "Monthly_Imports_Billion",
                        "Monthly_Exports_Billion",
                        "Monthly_Deficit_Billion",
                        "Monthly_Total_Trade_Billion"
                    ],
                    var_name="Indicator",
                    value_name="Rs_Billion"
                )

                fig_monthly = px.bar(
                    monthly_long,
                    x="Order",
                    y="Rs_Billion",
                    color="Indicator",
                    barmode="group",
                    title="Estimated Monthly Movement from Cumulative Data",
                    labels={
                        "Order": "Month Order",
                        "Rs_Billion": "Rs Billion",
                        "Indicator": "Indicator"
                    },
                    hover_data=["Period"]
                )

                fig_monthly.update_layout(
                    height=540,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    title_font_size=22,
                    font=dict(color="#0F172A"),
                    xaxis=dict(gridcolor="rgba(15,23,42,0.08)"),
                    yaxis=dict(gridcolor="rgba(15,23,42,0.08)")
                )

                st.plotly_chart(fig_monthly, use_container_width=True)

                st.markdown("### Latest Movement Signal")

                if import_change > export_change:
                    movement_note = "Imports increased more than exports in the latest period, which may widen trade-deficit pressure."
                elif export_change > import_change:
                    movement_note = "Exports increased more than imports in the latest period, which may slightly ease trade-deficit pressure."
                else:
                    movement_note = "Imports and exports moved by a similar amount in the latest period."

                st.markdown(
                    f"""
                    <div class="insight-card">
                        <p>{movement_note}</p>
                        <p>
                        Latest cumulative imports changed by <b>Rs {import_change:,.2f} billion</b>,
                        while cumulative exports changed by <b>Rs {export_change:,.2f} billion</b>.
                        The trade deficit changed by <b>Rs {deficit_change:,.2f} billion</b>.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                st.markdown("### Trend Insight Summary")

                st.markdown(
                    f"""
                    <div class="insight-card">
                        <h3>What changed in the latest period?</h3>
                        <p>{movement_note}</p>
                        <p>
                        Across <b>{len(trend_df)}</b> monthly customs files, the latest period is
                        <b>{latest['Period']}</b>. Cumulative imports reached
                        <b>Rs {latest['Imports_Billion']:,.2f} billion</b>, exports reached
                        <b>Rs {latest['Exports_Billion']:,.2f} billion</b>, and the trade deficit stood at
                        <b>Rs {latest['Trade_Deficit_Billion']:,.2f} billion</b>.
                        </p>
                        <p>
                        Compared with the previous cumulative file, imports changed by
                        <b>Rs {import_change:,.2f} billion</b>, exports changed by
                        <b>Rs {export_change:,.2f} billion</b>, and the trade deficit changed by
                        <b>Rs {deficit_change:,.2f} billion</b>.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("### Trend Data Table")

                show_cols = [
                    "Order",
                    "Period",
                    "Imports_Billion",
                    "Exports_Billion",
                    "Trade_Deficit_Billion",
                    "Total_Trade_Billion",
                    "Monthly_Imports_Billion",
                    "Monthly_Exports_Billion",
                    "Monthly_Deficit_Billion",
                    "Monthly_Total_Trade_Billion",
                    "Import_Export_Ratio"
                ]

                st.dataframe(
                    trend_df[show_cols].style.format({
                        "Imports_Billion": "{:,.2f}",
                        "Exports_Billion": "{:,.2f}",
                        "Trade_Deficit_Billion": "{:,.2f}",
                        "Total_Trade_Billion": "{:,.2f}",
                        "Monthly_Imports_Billion": "{:,.2f}",
                        "Monthly_Exports_Billion": "{:,.2f}",
                        "Monthly_Deficit_Billion": "{:,.2f}",
                        "Monthly_Total_Trade_Billion": "{:,.2f}",
                        "Import_Export_Ratio": "{:,.1f}"
                    }),
                    use_container_width=True
                )

                st.download_button(
                    "Download trend data as CSV",
                    data=trend_df[show_cols].to_csv(index=False).encode("utf-8"),
                    file_name="tradepulse_multi_month_trends.csv",
                    mime="text/csv"
                )


                # -----------------------------
                # Product Movement Over Time
                # -----------------------------

                st.markdown("---")
                st.markdown("### Product Movement Over Time")

                st.markdown(
                    """
                    <div class="insight-card">
                        <b>How this works:</b> The monthly customs files are cumulative, so latest monthly
                        product movement is estimated by subtracting the previous cumulative file from the latest cumulative file.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                try:
                    import_product_movement = build_product_movement_table(
                        trend_files=trend_files,
                        sheet_name="5_Imports_By_Commodity",
                        value_col="Imports_Value"
                    )

                    export_product_movement = build_product_movement_table(
                        trend_files=trend_files,
                        sheet_name="7_Exports_By_Commodity",
                        value_col="Exports_Value"
                    )

                    min_latest_value = st.number_input(
                        "Minimum latest monthly product value, Rs billion",
                        min_value=0.0,
                        value=0.1,
                        step=0.1,
                        key="product_movement_min_value"
                    )

                    top_product_n = st.slider(
                        "Number of products to show",
                        min_value=5,
                        max_value=30,
                        value=10,
                        step=5,
                        key="product_movement_top_n"
                    )

                    filtered_import_movement = import_product_movement[
                        import_product_movement["Latest_Month_Billion"] >= min_latest_value
                    ].copy()

                    filtered_export_movement = export_product_movement[
                        export_product_movement["Latest_Month_Billion"] >= min_latest_value
                    ].copy()

                    rising_imports = (
                        filtered_import_movement
                        .sort_values("Change_Billion", ascending=False)
                        .head(top_product_n)
                    )

                    falling_imports = (
                        filtered_import_movement
                        .sort_values("Change_Billion", ascending=True)
                        .head(top_product_n)
                    )

                    rising_exports = (
                        filtered_export_movement
                        .sort_values("Change_Billion", ascending=False)
                        .head(top_product_n)
                    )

                    falling_exports = (
                        filtered_export_movement
                        .sort_values("Change_Billion", ascending=True)
                        .head(top_product_n)
                    )

                    st.markdown("#### Top Product Movement Signals")

                    m1, m2, m3, m4 = st.columns(4)

                    with m1:
                        if len(rising_imports) > 0:
                            kpi_card(
                                "Top Rising Import",
                                short_text(rising_imports.iloc[0]["Description"], 22),
                                f"+Rs {rising_imports.iloc[0]['Change_Billion']:,.2f}B"
                            )

                    with m2:
                        if len(falling_imports) > 0:
                            kpi_card(
                                "Top Falling Import",
                                short_text(falling_imports.iloc[0]["Description"], 22),
                                f"Rs {falling_imports.iloc[0]['Change_Billion']:,.2f}B"
                            )

                    with m3:
                        if len(rising_exports) > 0:
                            kpi_card(
                                "Top Rising Export",
                                short_text(rising_exports.iloc[0]["Description"], 22),
                                f"+Rs {rising_exports.iloc[0]['Change_Billion']:,.2f}B"
                            )

                    with m4:
                        if len(falling_exports) > 0:
                            kpi_card(
                                "Top Falling Export",
                                short_text(falling_exports.iloc[0]["Description"], 22),
                                f"Rs {falling_exports.iloc[0]['Change_Billion']:,.2f}B"
                            )

                    st.markdown("### Rising and Falling Imports")

                    im1, im2 = st.columns(2)

                    with im1:
                        fig_rising_imports = horizontal_bar(
                            rising_imports,
                            "Change_Billion",
                            "Description",
                            "Top Rising Imports",
                            "Change vs Previous Month, Rs Billion",
                            "Product",
                            height=520
                        )
                        st.plotly_chart(fig_rising_imports, use_container_width=True)

                    with im2:
                        fig_falling_imports = horizontal_bar(
                            falling_imports,
                            "Change_Billion",
                            "Description",
                            "Top Falling Imports",
                            "Change vs Previous Month, Rs Billion",
                            "Product",
                            height=520
                        )
                        st.plotly_chart(fig_falling_imports, use_container_width=True)

                    st.markdown("### Rising and Falling Exports")

                    ex1, ex2 = st.columns(2)

                    with ex1:
                        fig_rising_exports = horizontal_bar(
                            rising_exports,
                            "Change_Billion",
                            "Description",
                            "Top Rising Exports",
                            "Change vs Previous Month, Rs Billion",
                            "Product",
                            height=520
                        )
                        st.plotly_chart(fig_rising_exports, use_container_width=True)

                    with ex2:
                        fig_falling_exports = horizontal_bar(
                            falling_exports,
                            "Change_Billion",
                            "Description",
                            "Top Falling Exports",
                            "Change vs Previous Month, Rs Billion",
                            "Product",
                            height=520
                        )
                        st.plotly_chart(fig_falling_exports, use_container_width=True)

                    st.markdown("### Product Movement Tables")

                    movement_cols = [
                        "HSCode",
                        "Description",
                        "Previous_Month_Billion",
                        "Latest_Month_Billion",
                        "Change_Billion",
                        "Growth_Percent",
                        "Latest_Cumulative_Billion"
                    ]

                    table1, table2 = st.columns(2)

                    with table1:
                        st.markdown("#### Import Movement Table")
                        st.dataframe(
                            rising_imports[movement_cols].style.format({
                                "Previous_Month_Billion": "{:,.2f}",
                                "Latest_Month_Billion": "{:,.2f}",
                                "Change_Billion": "{:,.2f}",
                                "Growth_Percent": "{:,.1f}",
                                "Latest_Cumulative_Billion": "{:,.2f}"
                            }),
                            use_container_width=True
                        )

                        st.download_button(
                            "Download import product movement",
                            data=import_product_movement.to_csv(index=False).encode("utf-8"),
                            file_name="tradepulse_import_product_movement.csv",
                            mime="text/csv"
                        )

                    with table2:
                        st.markdown("#### Export Movement Table")
                        st.dataframe(
                            rising_exports[movement_cols].style.format({
                                "Previous_Month_Billion": "{:,.2f}",
                                "Latest_Month_Billion": "{:,.2f}",
                                "Change_Billion": "{:,.2f}",
                                "Growth_Percent": "{:,.1f}",
                                "Latest_Cumulative_Billion": "{:,.2f}"
                            }),
                            use_container_width=True
                        )

                        st.download_button(
                            "Download export product movement",
                            data=export_product_movement.to_csv(index=False).encode("utf-8"),
                            file_name="tradepulse_export_product_movement.csv",
                            mime="text/csv"
                        )

                    st.markdown("### Product Movement Interpretation")

                    if len(rising_imports) > 0 and len(rising_exports) > 0:
                        top_rising_import_name = rising_imports.iloc[0]["Description"]
                        top_rising_import_change = rising_imports.iloc[0]["Change_Billion"]
                        top_rising_export_name = rising_exports.iloc[0]["Description"]
                        top_rising_export_change = rising_exports.iloc[0]["Change_Billion"]

                        st.markdown(
                            f"""
                            <div class="insight-card">
                                <p>
                                The strongest import increase in the latest month came from
                                <b>{top_rising_import_name}</b>, rising by around
                                <b>Rs {top_rising_import_change:,.2f} billion</b> compared with the previous monthly movement.
                                </p>
                                <p>
                                The strongest export increase came from
                                <b>{top_rising_export_name}</b>, rising by around
                                <b>Rs {top_rising_export_change:,.2f} billion</b>.
                                </p>
                                <p>
                                These movements can be used to identify emerging demand, trade shocks,
                                export momentum, import-substitution candidates, or products that deserve deeper investigation.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                    st.markdown("### Combined Trend Analyst Summary")

                    if len(rising_imports) > 0 and len(rising_exports) > 0:
                        top_rising_import_name = rising_imports.iloc[0]["Description"]
                        top_rising_import_change = rising_imports.iloc[0]["Change_Billion"]
                        top_rising_export_name = rising_exports.iloc[0]["Description"]
                        top_rising_export_change = rising_exports.iloc[0]["Change_Billion"]

                        if len(falling_imports) > 0:
                            top_falling_import_name = falling_imports.iloc[0]["Description"]
                            top_falling_import_change = falling_imports.iloc[0]["Change_Billion"]
                        else:
                            top_falling_import_name = "Not available"
                            top_falling_import_change = 0

                        if len(falling_exports) > 0:
                            top_falling_export_name = falling_exports.iloc[0]["Description"]
                            top_falling_export_change = falling_exports.iloc[0]["Change_Billion"]
                        else:
                            top_falling_export_name = "Not available"
                            top_falling_export_change = 0

                        st.markdown(
                            f"""
                            <div class="opportunity-card">
                                <h3>TradePulse Trend Read</h3>
                                <p>
                                The latest monthly movement shows where Nepal's trade flow is changing most sharply.
                                The strongest import rise came from <b>{top_rising_import_name}</b>
                                with a change of <b>Rs {top_rising_import_change:,.2f} billion</b>.
                                The strongest export rise came from <b>{top_rising_export_name}</b>
                                with a change of <b>Rs {top_rising_export_change:,.2f} billion</b>.
                                </p>
                                <p>
                                On the downside, the largest import slowdown was linked to
                                <b>{top_falling_import_name}</b> with a change of
                                <b>Rs {top_falling_import_change:,.2f} billion</b>, while the largest export slowdown was linked to
                                <b>{top_falling_export_name}</b> with a change of
                                <b>Rs {top_falling_export_change:,.2f} billion</b>.
                                </p>
                                <p>
                                These product movements can help identify emerging demand, seasonal shocks,
                                export momentum, import-substitution candidates, and possible media or research story angles.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                except Exception as e:
                    st.warning(f"Could not generate product movement analysis: {e}")

# -----------------------------
# About / Methodology tab
# -----------------------------

with about_tab:
    st.subheader("About TradePulse Nepal")

    st.markdown(
        """
        <div class="section-card">
            <h3>What is TradePulse Nepal?</h3>
            <p>
            TradePulse Nepal is a data dashboard that converts Nepal's Department of Customs trade data
            into simple market intelligence. It helps users understand imports, exports, trade deficit,
            product movement, country dependence, customs route concentration, business opportunities,
            and policy risks.
            </p>
            <p>
            The goal is to make public trade data easier to understand for students, researchers,
            journalists, businesses, policymakers, and analysts.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Data Status and Trust Notes")

    dsn1, dsn2, dsn3 = st.columns(3)
    with dsn1:
        st.metric("Current period", current_period_display)
    with dsn2:
        st.metric("Monthly files loaded", monthly_files_count)
    with dsn3:
        st.metric("Latest monthly file", latest_month_label)

    st.info(
        "TradePulse Nepal is a free public-data project. It uses Department of Customs workbooks available to the app. "
        "The dashboard does not modify official data; it cleans, converts, ranks, and explains the numbers for easier reading."
    )

    m1, m2 = st.columns(2)

    with m1:
        st.markdown(
            """
            <div class="section-card">
                <h3>Data Source</h3>
                <p><b>Primary source:</b> Department of Customs, Government of Nepal.</p>
                <p><b>Dataset type:</b> Monthly foreign trade statistics.</p>
                <p><b>Coverage:</b> Imports, exports, partner countries, commodities, HS codes, customs offices, quantity, value, and revenue.</p>
                <p><b>Current dashboard period:</b> shown in the Data Status section on top of the dashboard.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m2:
        st.markdown(
            """
            <div class="section-card">
                <h3>Units and Conversion</h3>
                <p>The original customs workbook reports many monetary values in <b>Rs. thousands</b>.</p>
                <p>This dashboard converts values into <b>Rs. billion</b> for easier reading.</p>
                <p><b>Formula used:</b></p>
                <p>Rs. billion = Source value ÷ 1,000,000</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Methodology")

    st.markdown(
        """
        <div class="section-card">
            <h3>How the Dashboard Works</h3>
            <ol>
                <li>The user uploads a Department of Customs Excel workbook.</li>
                <li>The dashboard reads trade, product, country, partner, export, import, and customs-office sheets.</li>
                <li>Values are cleaned and converted from Rs. thousands to Rs. billion.</li>
                <li>Total rows are removed so rankings show actual products, countries, and customs routes.</li>
                <li>The dashboard calculates key indicators such as total imports, exports, trade deficit, import-export ratio, country shares, product rankings, and route concentration.</li>
                <li>The Opportunity Finder ranks products using import value, customs revenue, and approximate duty signal.</li>
                <li>The Insights tab generates a simple trade brief using calculated dashboard numbers.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Opportunity Finder Score")

    st.markdown(
        """
        <div class="opportunity-card">
            <p><b>The Opportunity Finder is a screening tool, not an investment recommendation.</b></p>
            <p>It ranks imported products using three signals:</p>
            <ul>
                <li><b>Market Size Score:</b> based on import value.</li>
                <li><b>Revenue Signal Score:</b> based on customs revenue.</li>
                <li><b>Duty Signal Score:</b> based on approximate duty/revenue rate.</li>
            </ul>
            <p>The score helps identify products that may deserve deeper research for import-substitution,
            sourcing, distribution, trade finance, or policy analysis.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Limitations")

    st.markdown(
        """
        <div class="risk-card">
            <ul>
                <li>The dashboard depends on the structure and accuracy of the uploaded customs workbook.</li>
                <li>The Opportunity Score does not prove profitability or feasibility.</li>
                <li>Import-substitution potential requires additional data on domestic production, demand, costs, technology, and regulation.</li>
                <li>AI-style insights are generated only from available dashboard numbers and should be verified before publication.</li>
                <li>This dashboard is for research, business intelligence, and policy discussion — not financial or investment advice.</li>
                <li>Users should verify figures with the original Department of Customs workbook before citing them formally.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Suggested Citation")

    st.code(
        "TradePulse Nepal Dashboard. Based on monthly foreign trade statistics published by the Department of Customs, Government of Nepal.",
        language="text"
    )

    st.markdown("### Feedback and Contact")

    st.markdown(
        '<div class="feedback-card">'
        '<h3>Help improve TradePulse Nepal</h3>'
        '<p>TradePulse Nepal is a free public-data project. Feedback is welcome, especially if you find a data issue, broken chart, confusing label, missing feature, or a useful trade question the dashboard should answer.</p>'
        '<div class="contact-grid">'
        '<div class="contact-item"><b>Email</b>utsavkphuyal@gmail.com</div>'
        '<div class="contact-item"><b>LinkedIn</b>linkedin.com/in/utsav-phuyal</div>'
        '<div class="contact-item"><b>GitHub</b>github.com/utsavhatescoding</div>'
        '</div>'
        '<p><b>Suggested feedback:</b> missing products, wrong labels, monthly data update issues, UI problems, or questions Ask TradePulse should answer better.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.subheader("Developer")

    dev_col1, dev_col2 = st.columns([0.32, 0.68])

    with dev_col1:
        if developer_photo_path.exists():
            st.image(str(developer_photo_path), caption="Utsav Phuyal", use_container_width=True)
        else:
            st.info("Upload your photo as utsav.png in the same folder as app.py to show it here.")

    with dev_col2:
        st.markdown("### Utsav Phuyal")
        st.markdown("**Developer & Researcher, TradePulse Nepal**")

        st.write(
            "I am a business and economics graduate student interested in data analytics, "
            "economic research, trade intelligence, financial stability, and AI-powered "
            "public-data tools."
        )

        st.write(
            "I built TradePulse Nepal to make Nepal's Department of Customs data easier "
            "to understand through dashboards, product-level analysis, country intelligence, "
            "opportunity signals, and automated trade briefs."
        )

        st.write(
            "The goal is to turn raw public data into clear market insights, business signals, "
            "policy risks, and report-ready analysis."
        )

        st.markdown("**Contact**")
        st.markdown(
            "Email: utsavkphuyal@gmail.com  \n"
            "LinkedIn: linkedin.com/in/utsav-phuyal  \n"
            "GitHub: github.com/utsavhatescoding"
        )
# -----------------------------
# Insights tab
# -----------------------------

with insight_tab:
    st.subheader("Automated Trade Analyst")

    st.markdown(
        """
        <div class="insight-card">
            <b>What changed in this version:</b> this section now works like a rule-based analyst.
            It reads the dashboard's calculated numbers and turns them into simple explanations, opportunity signals,
            and risk signals. No paid AI API is used yet.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Build HTML with no leading spaces. Streamlit/Markdown can treat indented
    # HTML as a code block, which makes raw <div> text appear on screen.
    card_parts = ['<div class="analyst-grid">']
    for item in automated_insights["cards"]:
        card_parts.append(
            f'<div class="analyst-card">'
            f'<div class="insight-badge">{clean_display(item["badge"])}</div>'
            f'<h4>{clean_display(item["title"])}</h4>'
            f'<p>{clean_display(item["body"])}</p>'
            f'</div>'
        )
    card_parts.append('</div>')
    card_html = "".join(card_parts)

    st.markdown(card_html, unsafe_allow_html=True)

    brief_text = automated_insights["brief_text"]
    brief_html = clean_display(brief_text).replace("\n", "<br>")

    st.markdown("### Executive Brief")

    st.markdown(
        f"""
        <div class="executive-brief">
            {brief_html}
        </div>
        """,
        unsafe_allow_html=True
    )

    i1, i2 = st.columns(2)

    with i1:
        opportunity_items = "".join(
            [f"<li>{clean_display(item)}</li>" for item in automated_insights["opportunities"]]
        )

        st.markdown(
            f"""
            <div class="opportunity-card">
                <h3>Opportunity Signals</h3>
                <ul>{opportunity_items}</ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with i2:
        risk_items = "".join(
            [f"<li>{clean_display(item)}</li>" for item in automated_insights["risks"]]
        )

        st.markdown(
            f"""
            <div class="risk-card">
                <h3>Risk Signals</h3>
                <ul>{risk_items}</ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Media / Research Story Ideas")

    story_ideas = [
        f"Why Nepal's imports are still around {import_export_ratio:,.1f} times larger than exports.",
        f"What {top_import_product} tells us about Nepal's import demand.",
        f"How {top_export_product} became a leading export item in the current period.",
        f"What Nepal's dependence on {top_import_country} means for trade vulnerability.",
        f"Why customs route concentration around {top_customs_office} matters for logistics."
    ]

    st.markdown("\n".join([f"{idx + 1}. {idea}" for idx, idea in enumerate(story_ideas)]))

    st.markdown("---")
    st.markdown("## Download Report")

    st.markdown(
        """
        <div class="insight-card">
            Download the current TradePulse Nepal brief as a simple text file or as a professionally formatted PDF report.
            The PDF includes the market snapshot, product signals, country/customs movement, policy risks, business opportunities,
            and multi-month trend summary where monthly data is available.
        </div>
        """,
        unsafe_allow_html=True
    )

    trend_summary_for_pdf = build_trend_summary_for_report(monthly_data_path, selected_trend_end_name)

    pdf_buffer = create_pdf_report(
        current_col=current_col,
        imports_total=imports_total,
        exports_total=exports_total,
        deficit_total=deficit_total,
        total_trade=total_trade,
        import_export_ratio=import_export_ratio,
        top_import_product=top_import_product,
        top_import_value=top_import_value,
        top_export_product=top_export_product,
        top_export_value=top_export_value,
        top_import_country=top_import_country,
        top_export_country=top_export_country,
        top_customs_office=top_customs_office,
        top_customs_value=top_customs_value,
        trend_summary=trend_summary_for_pdf
    )

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            label="Download TXT Brief",
            data=brief_text.encode("utf-8"),
            file_name="tradepulse_nepal_monthly_brief.txt",
            mime="text/plain",
            key="download_txt_brief_main"
        )

    with d2:
        st.download_button(
            label="Download PDF Brief",
            data=pdf_buffer,
            file_name="tradepulse_nepal_monthly_brief.pdf",
            mime="application/pdf",
            key="download_pdf_brief_main"
        )

    st.caption("PDF button is in this Download Report section inside the Insights tab.")


# -----------------------------
# Ask TradePulse tab
# -----------------------------

with ask_tab:
    st.subheader("Ask TradePulse")

    st.info(
        "This is a free rule-based question assistant. It answers only from the processed Customs data in this dashboard. "
        "It does not use the internet, paid AI, or forecasting."
    )

    suggested_questions = [
        "What changed in the latest month?",
        "Where are the business opportunities?",
        "What are the main trade risks?",
        "Which country dominates imports?",
        "Which product increased the most?",
        "What are the top import products?",
        "What are the top export products?",
        "What is the trade deficit?",
        "Which customs route is most important?",
        "Give me a short trade brief.",
        "Give me a media story idea.",
        "What is Nepal importing most?",
        "What is Nepal exporting most?"
    ]

    if "ask_tradepulse_question_text" not in st.session_state:
        st.session_state["ask_tradepulse_question_text"] = "What changed in the latest month?"

    if "ask_tradepulse_answer" not in st.session_state:
        st.session_state["ask_tradepulse_answer"] = ""

    st.markdown("#### Suggested questions")
    st.caption("Click one question, then press Ask TradePulse.")

    suggestion_cols = st.columns(3)
    for i, q_text in enumerate(suggested_questions):
        with suggestion_cols[i % 3]:
            if st.button(q_text, key=f"ask_suggested_question_{i}", use_container_width=True):
                st.session_state["ask_tradepulse_question_text"] = q_text
                st.session_state["ask_tradepulse_answer"] = ""
                st.rerun()

    st.markdown("---")

    user_question = st.text_area(
        "Ask a question",
        height=100,
        placeholder="Example: What changed in the latest month?",
        key="ask_tradepulse_question_text"
    )

    if st.button("Ask TradePulse", type="primary", key="ask_tradepulse_button"):
        answer = ask_tradepulse_rule_based(
            question=user_question,
            imports_total=imports_total,
            exports_total=exports_total,
            deficit_total=deficit_total,
            total_trade=total_trade,
            import_export_ratio=import_export_ratio,
            top_import_product=top_import_product,
            top_import_value=top_import_value,
            top_export_product=top_export_product,
            top_export_value=top_export_value,
            top_import_country=top_import_country,
            top_export_country=top_export_country,
            top_customs_office=top_customs_office,
            top_customs_value=top_customs_value,
            top5_route_share=top5_route_share,
            imports=imports,
            exports=exports,
            countries=countries,
            customs=customs,
            monthly_data_path=monthly_data_path
        )
        st.session_state["ask_tradepulse_answer"] = answer

    if st.session_state.get("ask_tradepulse_answer"):
        st.markdown("### Answer")
        st.markdown(st.session_state["ask_tradepulse_answer"])

        answer_text = st.session_state["ask_tradepulse_answer"]
        question_text = st.session_state.get("ask_tradepulse_question_text", "")
        download_text = f"TradePulse Nepal - Ask TradePulse Answer\n\nQuestion:\n{question_text}\n\nAnswer:\n{answer_text}\n"

        a1, a2 = st.columns([1, 1])
        with a1:
            st.download_button(
                label="Download answer as TXT",
                data=download_text.encode("utf-8"),
                file_name="ask_tradepulse_answer.txt",
                mime="text/plain",
                key="download_ask_tradepulse_answer"
            )
        with a2:
            with st.expander("Copy-ready text"):
                st.text_area(
                    "Copy this answer",
                    value=download_text,
                    height=220,
                    key="copy_ready_ask_tradepulse_answer"
                )

    st.markdown("---")
    st.markdown("### What this version can answer")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            **Market**

            - Trade deficit
            - Import-export ratio
            - Latest movement
            """
        )

    with c2:
        st.markdown(
            """
            **Products & countries**

            - Top imports
            - Top exports
            - Import/export partners
            """
        )

    with c3:
        st.markdown(
            """
            **Signals**

            - Risks
            - Opportunities
            - Customs routes
            """
        )

    st.caption(
        "Later, this tab can be upgraded into a real AI chat interface using an API key. "
        "For now, it is intentionally simple, free, and safer."
    )





# -----------------------------
# Gemini AI Analyst tab
# -----------------------------

with gemini_tab:
    st.subheader("Gemini AI Analyst")

    st.info(
        "This is an optional Google Gemini-based analyst. The stable rule-based Ask TradePulse tab remains the default. "
        "Gemini runs only after you click the button. It receives a rich processed context with product movement, sectors, partners, customs routes, and monthly trends, but not raw Excel files."
    )

    api_key = st.secrets.get("GEMINI_API_KEY", "")

    if not api_key:
        st.warning("Gemini API key is not set yet. Add GEMINI_API_KEY in Streamlit Secrets to activate this tab.")
        with st.expander("How to set up Gemini API key"):
            st.markdown(
                """
                1. Create a Gemini API key from Google AI Studio.
                2. In Streamlit Cloud, open your app settings.
                3. Go to **Secrets**.
                4. Add:

                ```toml
                GEMINI_API_KEY = "paste_your_key_here"
                ```

                5. Add `google-genai` to `requirements.txt`.
                6. Reboot/redeploy the app.
                """
            )

    default_ai_question = "Give me an analyst brief on Nepal's latest trade situation."
    ai_question = st.text_area(
        "Ask the Gemini analyst",
        value=default_ai_question,
        height=110,
        placeholder="Example: What are the main risks and opportunities in the latest customs data?",
        key="gemini_ai_question"
    )

    g1, g2 = st.columns([1, 1])
    with g1:
        model_name = st.selectbox(
            "Gemini model",
            ["gemini-3.1-flash-lite", "gemini-2.0-flash-lite", "gemini-2.0-flash"],
            index=0,
            help="Use Flash-Lite first. It is lighter and safer for Streamlit Cloud. If one model fails, try another available model."
        )
    with g2:
        context_mode = st.selectbox(
            "AI data context",
            ["Detailed", "Balanced", "Light"],
            index=0,
            help="Detailed gives Gemini more processed product, sector, partner, customs, and movement data. Use Light only if the app is slow."
        )
        st.metric("Data sent to AI", context_mode, "Processed data only")

    st.caption(
        "For stability on Streamlit Cloud, Gemini context is prepared only after you click the button. "
        "This prevents slow page loads while still giving Gemini a rich processed context."
    )

    if "gemini_answer" not in st.session_state:
        st.session_state["gemini_answer"] = ""

    if st.button("Generate Gemini AI analysis", type="primary", key="generate_gemini_analysis"):
        data_context = build_compact_gemini_context(
            current_col=current_col,
            source_file_label=source_file_label,
            latest_month_label=latest_month_label,
            monthly_files_count=monthly_files_count,
            imports_total=imports_total,
            exports_total=exports_total,
            deficit_total=deficit_total,
            total_trade=total_trade,
            import_export_ratio=import_export_ratio,
            imports=imports,
            exports=exports,
            countries=countries,
            customs=customs,
            sector_summary=sector_summary,
            monthly_data_path=monthly_data_path,
            context_mode=context_mode,
            trend_end_file_name=selected_trend_end_name,
        )

        with st.spinner("Gemini is reading the processed TradePulse context..."):
            ai_answer = generate_gemini_trade_answer(
                question=ai_question,
                data_context=data_context,
                api_key=api_key,
                model_name=model_name
            )
        st.session_state["gemini_answer"] = ai_answer

    if st.session_state.get("gemini_answer"):
        st.markdown("### Gemini answer")
        st.markdown(st.session_state["gemini_answer"])

        gemini_download = (
            f"TradePulse Nepal - Gemini AI Analyst\n\n"
            f"Question:\n{ai_question}\n\n"
            f"Answer:\n{st.session_state['gemini_answer']}\n"
        )
        st.download_button(
            label="Download Gemini answer as TXT",
            data=gemini_download.encode("utf-8"),
            file_name="tradepulse_gemini_answer.txt",
            mime="text/plain",
            key="download_gemini_answer"
        )

    st.caption(
        "Gemini output should be reviewed before use in formal reports or media articles. "
        "The safe rule-based Ask TradePulse tab remains the default free fallback."
    )


# -----------------------------
# Footer
# -----------------------------

st.markdown(
    '<div class="tp-footer">'
    '<b>TradePulse Nepal</b> · Free public data project · Built by Utsav Phuyal<br>'
    'Data source: Department of Customs, Government of Nepal · Values are converted for easier reading and should be verified with the original workbook before formal citation.<br>'
    'Contact: utsavkphuyal@gmail.com · GitHub: github.com/utsavhatescoding'
    '</div>',
    unsafe_allow_html=True
)
