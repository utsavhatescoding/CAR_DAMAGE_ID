import base64
import io

import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="CarDD Vision — Damage Scanner",
    page_icon="🛠️",
    layout="wide",
)

# =====================================================================
# DESIGN SYSTEM
# ---------------------------------------------------------------------
# Concept: an automotive diagnostic scanner. Dark instrument-panel
# background, amber "warning light" accent, monospace technical
# readouts, viewfinder brackets around imagery, ticket-style report.
# =====================================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg:        #0E1113;
    --panel:     #171B1E;
    --panel-2:   #1E2327;
    --line:      rgba(237,234,225,0.10);
    --paper:     #EDEAE1;
    --muted:     #8C949B;
    --amber:     #FFB627;
    --amber-dim: rgba(255,182,39,0.22);
    --red:       #D7263D;
    --red-dim:   rgba(215,38,61,0.16);
    --teal:      #2EC4B6;
    --teal-dim:  rgba(46,196,182,0.16);
}

/* ---------- base canvas ---------- */
[data-testid="stAppViewContainer"] {
    background:
        repeating-linear-gradient(0deg, transparent, transparent 39px, var(--line) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, var(--line) 40px),
        var(--bg);
    color: var(--paper);
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    color: var(--amber);
    text-transform: uppercase;
    margin: 0 0 0.35rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.eyebrow::before {
    content: "";
    width: 7px; height: 7px;
    background: var(--amber);
    display: inline-block;
    box-shadow: 0 0 6px var(--amber);
}

[data-testid="stWidgetLabel"] p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--muted) !important;
}

/* selectbox */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: var(--panel-2);
    border: 1px solid var(--line);
    border-radius: 3px;
    color: var(--paper);
}

/* slider */
[data-testid="stSlider"] [role="slider"] {
    background-color: var(--amber) !important;
    box-shadow: 0 0 0 4px var(--amber-dim);
}
[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
    background: var(--amber) !important;
}
[data-testid="stTickBar"] { display: none; }

/* buttons */
.stButton > button {
    background: var(--amber);
    color: #171100;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-size: 0.82rem;
    border: none;
    border-radius: 2px;
    padding: 0.85rem 1rem;
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}
.stButton > button:hover {
    box-shadow: 0 0 0 3px var(--amber-dim), 0 0 22px var(--amber-dim);
    transform: translateY(-1px);
    color: #171100;
}
.stButton > button:active { transform: translateY(0); }

/* file uploader */
[data-testid="stFileUploaderDropzone"] {
    background: var(--panel-2);
    border: 1.5px dashed rgba(255,182,39,0.35);
    border-radius: 4px;
}
[data-testid="stFileUploaderDropzone"] button {
    background: transparent;
    border: 1px solid var(--amber-dim);
    color: var(--amber);
}

/* spinner text */
[data-testid="stSpinner"] p {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--amber);
    letter-spacing: 0.04em;
}

hr, [data-testid="stDivider"] { border-color: var(--line) !important; }

/* ---------- hero ---------- */
.hero {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 1rem;
    padding-bottom: 1.4rem;
    margin-bottom: 1.8rem;
    border-bottom: 1px solid var(--line);
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: clamp(1.9rem, 4vw, 2.7rem);
    letter-spacing: -0.01em;
    margin: 0;
    line-height: 1.05;
}
.hero h1 span { color: var(--amber); }
.hero p.sub {
    font-family: 'Inter', sans-serif;
    color: var(--muted);
    margin: 0.4rem 0 0 0;
    font-size: 0.95rem;
}
.status-chip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.10em;
    color: var(--teal);
    border: 1px solid var(--teal-dim);
    background: rgba(46,196,182,0.06);
    padding: 0.42rem 0.75rem;
    border-radius: 2px;
    white-space: nowrap;
}
.status-chip .dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: var(--teal);
    border-radius: 50%;
    margin-right: 0.45rem;
    animation: pulse 1.8s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 var(--teal-dim); }
    50% { opacity: 0.55; box-shadow: 0 0 0 5px transparent; }
}

/* ---------- viewfinder image frame ---------- */
.frame-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.55rem;
}
.viewfinder {
    position: relative;
    padding: 12px;
    background: var(--panel);
    border: 1px solid var(--line);
    overflow: hidden;
}
.viewfinder img { width: 100%; display: block; border-radius: 1px; }
.corner { position: absolute; width: 20px; height: 20px; border-color: var(--amber); }
.corner-tl { top: 8px; left: 8px; border-top: 2.5px solid; border-left: 2.5px solid; }
.corner-tr { top: 8px; right: 8px; border-top: 2.5px solid; border-right: 2.5px solid; }
.corner-bl { bottom: 8px; left: 8px; border-bottom: 2.5px solid; border-left: 2.5px solid; }
.corner-br { bottom: 8px; right: 8px; border-bottom: 2.5px solid; border-right: 2.5px solid; }

.scanline {
    position: absolute;
    left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--amber), transparent);
    opacity: 0.75;
    animation: sweep 2.6s linear infinite;
}
@keyframes sweep {
    0% { top: 0%; }
    100% { top: 100%; }
}
@media (prefers-reduced-motion: reduce) {
    .scanline, .status-chip .dot { animation: none; }
}

/* ---------- damage report ticket ---------- */
.report-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1.15rem;
    margin: 0 0 0.9rem 0;
}
.ticket {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    background: var(--panel);
    border: 1px solid var(--line);
    border-left: 3px solid var(--red);
    padding: 0.75rem 1rem;
    margin-bottom: 0.55rem;
}
.ticket.clear { border-left-color: var(--teal); }
.ticket .name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    min-width: 190px;
}
.ticket .bar-track {
    flex: 1;
    height: 6px;
    background: var(--panel-2);
    border-radius: 3px;
    overflow: hidden;
    min-width: 60px;
}
.ticket .bar-fill { height: 100%; background: var(--red); }
.ticket .score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: var(--muted);
    min-width: 52px;
    text-align: right;
}
.severity {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    padding: 0.15rem 0.45rem;
    border-radius: 2px;
    min-width: 46px;
    text-align: center;
}
.sev-high { background: var(--red-dim); color: var(--red); }
.sev-med  { background: var(--amber-dim); color: var(--amber); }
.sev-low  { background: rgba(140,148,155,0.14); color: var(--muted); }

.clear-banner {
    font-family: 'IBM Plex Mono', monospace;
    background: var(--teal-dim);
    border: 1px solid rgba(46,196,182,0.35);
    color: var(--teal);
    padding: 0.9rem 1.1rem;
    letter-spacing: 0.04em;
    font-size: 0.85rem;
}

.footer-strip {
    margin-top: 2.4rem;
    padding-top: 1rem;
    border-top: 1px solid var(--line);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ---------- mobile ---------- */
@media (max-width: 768px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .hero { flex-direction: column; align-items: flex-start; }
    .ticket { flex-wrap: wrap; }
    .ticket .name { min-width: 100%; }
    .ticket .bar-track { order: 3; width: 100%; }
    .ticket .score { order: 4; }
}
</style>
""",
    unsafe_allow_html=True,
)


def image_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def render_frame(img: Image.Image, label: str, scanning: bool = False):
    scanline = '<div class="scanline"></div>' if scanning else ""
    st.markdown(
        f"""
        <div class="frame-label">{label}</div>
        <div class="viewfinder">
            <span class="corner corner-tl"></span>
            <span class="corner corner-tr"></span>
            <span class="corner corner-bl"></span>
            <span class="corner corner-br"></span>
            {scanline}
            <img src="{image_to_data_uri(img)}" />
        </div>
        """,
        unsafe_allow_html=True,
    )


def severity_for(score: float):
    if score >= 0.75:
        return "HIGH", "sev-high"
    if score >= 0.50:
        return "MED", "sev-med"
    return "LOW", "sev-low"


# =====================================================================
# MODEL LOADERS
# =====================================================================

@st.cache_resource
def load_yolov8():
    model_path = hf_hub_download(
        repo_id="abdullahg7/cardd-yolov8s",
        filename="v2.0/best.pt",
    )
    return YOLO(model_path)


@st.cache_resource
def load_yolo11():
    model_path = hf_hub_download(
        repo_id="harpreetsahota/car-dd-segmentation-yolov11",
        filename="best.pt",
    )
    return YOLO(model_path)


# =====================================================================
# HERO
# =====================================================================

st.markdown(
    """
    <div class="hero">
        <div>
            <h1>CarDD <span>Vision</span></h1>
            <p class="sub">Point a camera at damage. Two detection models, one verdict.</p>
        </div>
        <div class="status-chip"><span class="dot"></span>MODELS READY</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# SIDEBAR
# =====================================================================

st.sidebar.markdown('<p class="eyebrow">Scanner Settings</p>', unsafe_allow_html=True)

model_choice = st.sidebar.selectbox(
    "Detection model",
    ["YOLOv8s — Fast", "YOLO11 — High Accuracy"],
)

confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05,
)

st.sidebar.markdown(
    f"""
    <div style="margin-top:1.6rem;padding:0.9rem;background:var(--panel-2);
                border:1px solid var(--line);font-family:'IBM Plex Mono',monospace;
                font-size:0.72rem;color:var(--muted);line-height:1.7;">
        MODEL &nbsp;&nbsp;{model_choice.split(' — ')[0]}<br>
        MODE &nbsp;&nbsp;&nbsp;{'SPEED' if 'Fast' in model_choice else 'ACCURACY'}<br>
        THRESH &nbsp;{confidence:.2f}
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# UPLOAD
# =====================================================================

st.markdown('<p class="eyebrow">Step 01 — Load Image</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload a car image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        render_frame(image, "Input Feed", scanning=True)

    run = st.button("▶  Run Diagnostic Scan", use_container_width=True)

    if run:
        with st.spinner("Loading model and scanning for damage…"):

            model = load_yolov8() if model_choice == "YOLOv8s — Fast" else load_yolo11()

            results = model.predict(
                source=np.array(image),
                conf=confidence,
                verbose=False,
            )

            result_image = Image.fromarray(results[0].plot()[:, :, ::-1])

        with col2:
            render_frame(result_image, "Scan Output")

        st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
        st.markdown('<p class="eyebrow">Step 02 — Damage Report</p>', unsafe_allow_html=True)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            st.markdown('<p class="report-title">Findings</p>', unsafe_allow_html=True)

            for box in results[0].boxes:
                class_id = int(box.cls[0])
                score = float(box.conf[0])
                damage_type = model.names[class_id].replace("_", " ").title()
                label, sev_class = severity_for(score)

                st.markdown(
                    f"""
                    <div class="ticket">
                        <span class="severity {sev_class}">{label}</span>
                        <span class="name">{damage_type}</span>
                        <div class="bar-track">
                            <div class="bar-fill" style="width:{score*100:.0f}%;"></div>
                        </div>
                        <span class="score">{score:.1%}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="clear-banner">✓ NO DAMAGE DETECTED — VEHICLE CLEAR</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="footer-strip">
                Model: {model.names.__class__.__name__ and model_choice.split(' — ')[0]}
                &nbsp;·&nbsp; Threshold: {confidence:.2f}
                &nbsp;·&nbsp; CarDD Vision System
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        """
        <div style="margin-top:0.4rem;padding:2.2rem;text-align:center;
                    background:var(--panel);border:1px dashed var(--line);
                    color:var(--muted);font-family:'IBM Plex Mono',monospace;
                    font-size:0.8rem;letter-spacing:0.04em;">
            AWAITING INPUT — upload a photo above to begin the scan
        </div>
        """,
        unsafe_allow_html=True,
    )
