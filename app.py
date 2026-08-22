import base64
import io
from datetime import datetime

import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CarDD Vision — AI Damage Scanner",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #0E1113;
    --panel: #171B1E;
    --panel2: #1E2327;
    --line: rgba(237,234,225,0.10);
    --paper: #EDEAE1;
    --muted: #8C949B;
    --amber: #FFB627;
    --amber-dim: rgba(255,182,39,0.18);
    --red: #D7263D;
    --red-dim: rgba(215,38,61,0.14);
    --teal: #2EC4B6;
    --teal-dim: rgba(46,196,182,0.14);
}

/* APP */

[data-testid="stAppViewContainer"] {
    background:
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 39px,
            var(--line) 40px
        ),
        repeating-linear-gradient(
            90deg,
            transparent,
            transparent 39px,
            var(--line) 40px
        ),
        var(--bg);
    color: var(--paper);
}

[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu,
footer {
    visibility: hidden;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}


/* SIDEBAR */

[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}


/* TYPOGRAPHY */

.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.70rem;
    letter-spacing: 0.15em;
    color: var(--amber);
    text-transform: uppercase;
    margin: 0 0 0.5rem 0;
}

.eyebrow-dot {
    color: var(--amber);
    margin-right: 7px;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.3rem);
    font-weight: 700;
    letter-spacing: -0.04em;
    line-height: 1;
    margin: 0;
}

.hero-title span {
    color: var(--amber);
}

.hero-subtitle {
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 0.65rem;
}


/* STATUS */

.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.70rem;
    letter-spacing: 0.10em;
    color: var(--teal);
    border: 1px solid rgba(46,196,182,0.30);
    background: rgba(46,196,182,0.05);
    padding: 0.5rem 0.75rem;
    border-radius: 3px;
}

.status-dot {
    width: 7px;
    height: 7px;
    background: var(--teal);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--teal);
}


/* SIDEBAR MODEL CARDS */

.model-card {
    background: var(--panel2);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 0.9rem;
    margin-top: 0.7rem;
}

.model-name {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--paper);
}

.model-meta {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--muted);
    font-size: 0.65rem;
    line-height: 1.7;
    margin-top: 0.45rem;
}


/* SELECTBOX */

[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label {
    color: var(--muted) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.70rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}


/* BUTTONS */

.stButton > button {
    background: var(--amber);
    color: #171100;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-size: 0.80rem;
    border: none;
    border-radius: 3px;
    padding: 0.85rem 1rem;
}

.stButton > button:hover {
    background: var(--amber);
    color: #171100;
    box-shadow:
        0 0 0 3px var(--amber-dim),
        0 0 25px var(--amber-dim);
}


/* UPLOADER */

[data-testid="stFileUploaderDropzone"] {
    background: var(--panel);
    border: 1.5px dashed rgba(255,182,39,0.40);
    border-radius: 5px;
}

[data-testid="stFileUploaderDropzone"] button {
    background: transparent !important;
    border: 1px solid var(--amber) !important;
    color: var(--amber) !important;
}

[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {
    color: var(--muted) !important;
}

[data-testid="stFileUploaderDropzone"] svg {
    color: var(--amber) !important;
}


/* VIEWFINDER */

.frame-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.55rem;
}

.viewfinder {
    position: relative;
    padding: 10px;
    background: var(--panel);
    border: 1px solid var(--line);
    overflow: hidden;
    border-radius: 4px;
}

.viewfinder img {
    width: 100%;
    display: block;
    border-radius: 2px;
}

.corner {
    position: absolute;
    width: 20px;
    height: 20px;
    border-color: var(--amber);
    z-index: 2;
}

.corner-tl {
    top: 6px;
    left: 6px;
    border-top: 2px solid;
    border-left: 2px solid;
}

.corner-tr {
    top: 6px;
    right: 6px;
    border-top: 2px solid;
    border-right: 2px solid;
}

.corner-bl {
    bottom: 6px;
    left: 6px;
    border-bottom: 2px solid;
    border-left: 2px solid;
}

.corner-br {
    bottom: 6px;
    right: 6px;
    border-bottom: 2px solid;
    border-right: 2px solid;
}


/* READY BOX */

.ready-box {
    min-height: 260px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    background: var(--panel);
    border: 1px dashed var(--line);
    border-radius: 4px;
    padding: 2rem;
}

.ready-icon {
    font-size: 3rem;
    margin-bottom: 0.7rem;
}

.ready-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--paper);
}

.ready-text {
    color: var(--muted);
    font-size: 0.78rem;
    margin-top: 0.4rem;
}


/* METRICS */

.metric-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 1rem;
    text-align: center;
}

.metric-icon {
    font-size: 1.2rem;
}

.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 0.2rem;
    color: var(--paper);
}

.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.10em;
    color: var(--muted);
    text-transform: uppercase;
}


/* DAMAGE REPORT */

.report-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1.2rem;
    margin: 0.8rem 0;
}

.ticket {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    background: var(--panel);
    border: 1px solid var(--line);
    border-left: 3px solid var(--red);
    padding: 0.75rem 1rem;
    margin-bottom: 0.55rem;
    border-radius: 2px;
}

.ticket-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    text-transform: uppercase;
    min-width: 150px;
    color: var(--paper);
}

.bar-track {
    flex: 1;
    height: 6px;
    background: var(--panel2);
    border-radius: 3px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    background: var(--red);
}

.score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    min-width: 55px;
    text-align: right;
}

.confidence-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.60rem;
    padding: 0.2rem 0.4rem;
    border-radius: 2px;
    min-width: 42px;
    text-align: center;
}

.high {
    color: var(--teal);
    background: var(--teal-dim);
}

.medium {
    color: var(--amber);
    background: var(--amber-dim);
}

.low {
    color: var(--muted);
    background: rgba(140,148,155,0.12);
}


/* SUMMARY CARD */

.summary-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 1rem;
}

.summary-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    color: var(--paper);
    margin-bottom: 0.35rem;
}

.summary-text {
    color: var(--muted);
    font-size: 0.78rem;
    line-height: 1.6;
}


/* CLEAR RESULT */

.clear-banner {
    font-family: 'IBM Plex Mono', monospace;
    background: var(--teal-dim);
    border: 1px solid rgba(46,196,182,0.35);
    color: var(--teal);
    padding: 1rem;
    border-radius: 3px;
    line-height: 1.6;
}


/* DIVIDER */

hr {
    border-color: var(--line) !important;
}


/* FOOTER */

.footer-strip {
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--line);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    line-height: 1.8;
}


/* MOBILE */

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .ticket {
        flex-wrap: wrap;
    }

    .ticket-name {
        min-width: 100%;
    }

    .bar-track {
        flex-basis: 100%;
        width: 100%;
    }

    .score {
        min-width: auto;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def image_to_data_uri(img: Image.Image) -> str:
    """Convert a PIL image into a browser-displayable data URI."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def render_frame(img: Image.Image, label: str):
    """Display an image inside the diagnostic frame."""
    uri = image_to_data_uri(img)

    html = (
        '<div class="frame-label">'
        + label
        + '</div>'
        + '<div class="viewfinder">'
        + '<span class="corner corner-tl"></span>'
        + '<span class="corner corner-tr"></span>'
        + '<span class="corner corner-bl"></span>'
        + '<span class="corner corner-br"></span>'
        + f'<img src="{uri}" />'
        + '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


def confidence_level(score: float):
    if score >= 0.75:
        return "HIGH", "high"
    if score >= 0.50:
        return "MED", "medium"
    return "LOW", "low"


def load_selected_model(model_choice: str):
    if "YOLOv8" in model_choice:
        return load_yolov8()

    return load_yolo11()


# ============================================================
# MODEL LOADERS
# ============================================================

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


# ============================================================
# HERO
# ============================================================

hero_left, hero_right = st.columns([4, 1])

with hero_left:

    st.markdown(
        '<div class="eyebrow">'
        '<span class="eyebrow-dot">●</span>'
        'AI Automotive Diagnostics'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-title">'
        'CarDD <span>Vision</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'AI-powered vehicle damage detection & visual inspection.'
        '</div>',
        unsafe_allow_html=True,
    )

with hero_right:

    st.markdown(
        '<div style="text-align:right;margin-top:1rem;">'
        '<div class="status-chip">'
        '<span class="status-dot"></span>'
        'SYSTEM ONLINE'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    '<div class="eyebrow">'
    '<span class="eyebrow-dot">●</span>'
    'Scanner Settings'
    '</div>',
    unsafe_allow_html=True,
)

model_choice = st.sidebar.selectbox(
    "Detection model",
    [
        "⚡ YOLOv8s — Fast",
        "🎯 YOLO11 — Advanced",
    ],
)

confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05,
)

st.sidebar.markdown(
    '<div class="model-card">'
    '<div class="model-name">⚡ YOLOv8s</div>'
    '<div class="model-meta">'
    'SPEED OPTIMIZED<br>'
    'CarDD trained<br>'
    'Fast inference'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="model-card">'
    '<div class="model-name">🎯 YOLO11</div>'
    '<div class="model-meta">'
    'ADVANCED MODEL<br>'
    'CarDD trained<br>'
    'Instance segmentation'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.sidebar.link_button(
    "🔗 YOLOv8 — Hugging Face",
    "https://huggingface.co/abdullahg7/cardd-yolov8s",
)

st.sidebar.link_button(
    "🔗 YOLO11 — Hugging Face",
    "https://huggingface.co/harpreetsahota/car-dd-segmentation-yolov11",
)

st.sidebar.markdown(
    '<div style="'
    'margin-top:1rem;'
    'padding:0.8rem;'
    'background:#1E2327;'
    'border:1px solid rgba(237,234,225,0.10);'
    'font-family:IBM Plex Mono,monospace;'
    'font-size:0.65rem;'
    'color:#8C949B;'
    'line-height:1.8;'
    '">'
    f'MODEL&nbsp;&nbsp;&nbsp; {model_choice.split(" — ")[0]}<br>'
    f'THRESH&nbsp;&nbsp; {confidence:.2f}<br>'
    'ENGINE&nbsp;&nbsp;&nbsp; CarDD Vision'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="eyebrow">'
    '<span class="eyebrow-dot">●</span>'
    '01 — Vehicle Image'
    '</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)


# ============================================================
# NO IMAGE
# ============================================================

if uploaded_file is None:

    st.markdown(
        '<div class="ready-box">'
        '<div class="ready-icon">🚗</div>'
        '<div class="ready-title">Vehicle scanner ready</div>'
        '<div class="ready-text">'
        'Upload a vehicle image to begin AI inspection.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# IMAGE UPLOADED
# ============================================================

else:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown(
        '<div style="'
        'font-family:IBM Plex Mono,monospace;'
        'font-size:0.65rem;'
        'color:#2EC4B6;'
        'margin:0.6rem 0 0.8rem 0;'
        '">'
        '● IMAGE RECEIVED'
        '</div>',
        unsafe_allow_html=True,
    )

    input_col, output_col = st.columns(
        2,
        gap="large",
    )

    with input_col:

        render_frame(
            image,
            "📷 Input Feed",
        )

    with output_col:

        st.markdown(
            '<div class="frame-label">🤖 AI Analysis</div>'
            '<div class="ready-box">'
            '<div class="ready-icon">🔍</div>'
            '<div class="ready-title">Ready for scan</div>'
            '<div class="ready-text">'
            'Select your model and run the diagnostic.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    run_scan = st.button(
        "🔍  RUN DIAGNOSTIC SCAN",
        use_container_width=True,
    )

    # ========================================================
    # RUN MODEL
    # ========================================================

    if run_scan:

        start_time = datetime.now()

        with st.spinner(
            f"Running {model_choice.split(' — ')[0]}..."
        ):

            model = load_selected_model(model_choice)

            results = model.predict(
                source=np.array(image),
                conf=confidence,
                verbose=False,
            )

            plotted = results[0].plot()

            result_image = Image.fromarray(
                plotted[:, :, ::-1]
            )

        processing_time = (
            datetime.now() - start_time
        ).total_seconds()

        boxes = results[0].boxes

        # ====================================================
        # OUTPUT
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="eyebrow">'
            '<span class="eyebrow-dot">●</span>'
            '02 — Detection Output'
            '</div>',
            unsafe_allow_html=True,
        )

        result_col, summary_col = st.columns(
            2,
            gap="large",
        )

        with result_col:

            render_frame(
                result_image,
                "🎯 Annotated Detection",
            )

        with summary_col:

            st.markdown(
                '<div class="frame-label">📊 Scan Summary</div>',
                unsafe_allow_html=True,
            )

            if boxes is not None and len(boxes) > 0:

                detection_count = len(boxes)

                scores = [
                    float(box.conf[0])
                    for box in boxes
                ]

                average_confidence = (
                    sum(scores) / len(scores)
                )

                damage_types = []

                for box in boxes:

                    class_id = int(box.cls[0])

                    damage_name = (
                        model.names[class_id]
                        .replace("_", " ")
                        .title()
                    )

                    if damage_name not in damage_types:
                        damage_types.append(damage_name)

                metric_col1, metric_col2 = st.columns(2)

                with metric_col1:

                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-icon">⚠️</div>'
                        f'<div class="metric-value">'
                        f'{detection_count}'
                        f'</div>'
                        f'<div class="metric-label">'
                        f'Detections'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                with metric_col2:

                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-icon">🎯</div>'
                        f'<div class="metric-value">'
                        f'{average_confidence:.0%}'
                        f'</div>'
                        f'<div class="metric-label">'
                        f'Avg Confidence'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown(
                    '<div class="summary-card">'
                    '<div class="summary-title">'
                    '🔧 Damage Categories'
                    '</div>'
                    '<div class="summary-text">'
                    + ", ".join(damage_types)
                    + '</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    '<div class="clear-banner">'
                    '✓ NO DAMAGE DETECTED'
                    '<br><br>'
                    'No damage was identified above the '
                    'selected confidence threshold.'
                    '</div>',
                    unsafe_allow_html=True,
                )

        # ====================================================
        # DAMAGE REPORT
        # ====================================================

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            '<div class="eyebrow">'
            '<span class="eyebrow-dot">●</span>'
            '03 — Detection Report'
            '</div>',
            unsafe_allow_html=True,
        )

        if boxes is not None and len(boxes) > 0:

            st.markdown(
                '<div class="report-title">'
                '🔧 Detected Findings'
                '</div>',
                unsafe_allow_html=True,
            )

            for box in boxes:

                class_id = int(box.cls[0])
                score = float(box.conf[0])

                damage_type = (
                    model.names[class_id]
                    .replace("_", " ")
                    .title()
                )

                level, level_class = confidence_level(score)

                st.markdown(
                    '<div class="ticket">'
                    f'<span class="confidence-badge {level_class}">'
                    f'{level}'
                    '</span>'
                    f'<span class="ticket-name">'
                    f'{damage_type}'
                    '</span>'
                    '<div class="bar-track">'
                    f'<div class="bar-fill" '
                    f'style="width:{score * 100:.0f}%;">'
                    '</div>'
                    '</div>'
                    f'<span class="score">{score:.1%}</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )

        # ====================================================
        # DOWNLOAD
        # ====================================================

        st.markdown("<br>", unsafe_allow_html=True)

        output_buffer = io.BytesIO()

        result_image.save(
            output_buffer,
            format="PNG",
        )

        st.download_button(
            "⬇️ DOWNLOAD ANNOTATED IMAGE",
            data=output_buffer.getvalue(),
            file_name="cardd_vision_damage_scan.png",
            mime="image/png",
            use_container_width=True,
        )

        # ====================================================
        # SCAN INFORMATION
        # ====================================================

        scan_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        st.markdown(
            '<div class="footer-strip">'
            f'MODEL · {model_choice.split(" — ")[0]}'
            '&nbsp; · &nbsp;'
            f'THRESHOLD · {confidence:.2f}'
            '&nbsp; · &nbsp;'
            f'PROCESSING · {processing_time:.2f}s'
            '&nbsp; · &nbsp;'
            f'SCAN · {scan_time}'
            '<br>'
            '⚠️ Confidence represents model confidence, '
            'not physical damage severity.'
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer-strip" style="text-align:center;">'
    '🚗 CarDD Vision'
    '&nbsp; · &nbsp;'
    'AI-assisted vehicle damage detection'
    '&nbsp; · &nbsp;'
    'YOLO + CarDD'
    '</div>',
    unsafe_allow_html=True,
)
