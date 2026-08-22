```python
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
# DESIGN SYSTEM
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
    --blue: #4EA5FF;
}

/* ==========================================================
   APP
   ========================================================== */

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

html,
body,
[class*="css"] {
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}


/* ==========================================================
   SIDEBAR
   ========================================================== */

[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}


/* ==========================================================
   TYPOGRAPHY
   ========================================================== */

.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.70rem;
    letter-spacing: 0.15em;
    color: var(--amber);
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}

.eyebrow::before {
    content: "●";
    margin-right: 7px;
    text-shadow: 0 0 8px var(--amber);
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 700;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 0.5rem;
}

.hero-title span {
    color: var(--amber);
}

.hero-subtitle {
    color: var(--muted);
    font-size: 0.95rem;
    margin-bottom: 1.4rem;
}


/* ==========================================================
   STATUS
   ========================================================== */

.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.70rem;
    letter-spacing: 0.10em;
    color: var(--teal);
    border: 1px solid var(--teal-dim);
    background: rgba(46,196,182,0.05);
    padding: 0.45rem 0.75rem;
    border-radius: 3px;
}

.status-dot {
    width: 7px;
    height: 7px;
    background: var(--teal);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--teal);
}


/* ==========================================================
   CARDS
   ========================================================== */

.info-card {
    background: var(--panel);
    border: 1px solid var(--line);
    padding: 1rem;
    border-radius: 4px;
    height: 100%;
}

.info-card-icon {
    font-size: 1.4rem;
    margin-bottom: 0.4rem;
}

.info-card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
}

.info-card-text {
    color: var(--muted);
    font-size: 0.78rem;
    margin-top: 0.25rem;
}


/* ==========================================================
   BUTTONS
   ========================================================== */

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
    box-shadow:
        0 0 0 3px var(--amber-dim),
        0 0 25px var(--amber-dim);
    color: #171100;
}


/* ==========================================================
   UPLOADER
   ========================================================== */

[data-testid="stFileUploaderDropzone"] {
    background: var(--panel2);
    border: 1.5px dashed rgba(255,182,39,0.35);
    border-radius: 5px;
    padding: 1rem;
}

[data-testid="stFileUploaderDropzone"] button {
    background: transparent;
    border: 1px solid var(--amber);
    color: var(--amber);
}


/* ==========================================================
   VIEWFINDER
   ========================================================== */

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


/* ==========================================================
   METRICS
   ========================================================== */

.metric-card {
    background: var(--panel);
    border: 1px solid var(--line);
    padding: 1rem;
    text-align: center;
    border-radius: 4px;
}

.metric-icon {
    font-size: 1.2rem;
}

.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 0.25rem;
}

.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.10em;
    color: var(--muted);
    text-transform: uppercase;
}


/* ==========================================================
   DAMAGE REPORT
   ========================================================== */

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

.ticket .name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    text-transform: uppercase;
    min-width: 155px;
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
    min-width: 40px;
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


/* ==========================================================
   CLEAR
   ========================================================== */

.clear-banner {
    font-family: 'IBM Plex Mono', monospace;
    background: var(--teal-dim);
    border: 1px solid rgba(46,196,182,0.35);
    color: var(--teal);
    padding: 1rem;
    border-radius: 3px;
}


/* ==========================================================
   MODEL CARD
   ========================================================== */

.model-card {
    background: var(--panel);
    border: 1px solid var(--line);
    padding: 1rem;
    border-radius: 4px;
    margin-bottom: 0.7rem;
}

.model-name {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
}

.model-meta {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--muted);
    font-size: 0.65rem;
    line-height: 1.7;
    margin-top: 0.5rem;
}


/* ==========================================================
   FOOTER
   ========================================================== */

.footer-strip {
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--line);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .ticket {
        flex-wrap: wrap;
    }

    .ticket .name {
        min-width: 100%;
    }

    .bar-track {
        width: 100%;
        flex-basis: 100%;
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
# HELPERS
# ============================================================

def image_to_data_uri(img: Image.Image) -> str:
    """Convert PIL image to base64 data URI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def render_frame(img: Image.Image, label: str):
    """Render an image inside the diagnostic viewfinder."""
    st.markdown(
        f"""
        <div class="frame-label">{label}</div>

        <div class="viewfinder">

            <span class="corner corner-tl"></span>
            <span class="corner corner-tr"></span>
            <span class="corner corner-bl"></span>
            <span class="corner corner-br"></span>

            <img src="{image_to_data_uri(img)}" />

        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_level(score):
    if score >= 0.75:
        return "HIGH", "high"
    elif score >= 0.50:
        return "MED", "medium"
    return "LOW", "low"


def get_unique_damage(results, model):
    damages = []

    if results[0].boxes is None:
        return damages

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        damage = model.names[class_id].replace("_", " ").title()

        if damage not in damages:
            damages.append(damage)

    return damages


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
        """
        <div class="eyebrow">AI Automotive Diagnostics</div>

        <div class="hero-title">
            CarDD <span>Vision</span>
        </div>

        <div class="hero-subtitle">
            AI-powered vehicle damage detection & visual inspection.
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_right:

    st.markdown(
        """
        <div style="text-align:right;margin-top:1rem;">
            <div class="status-chip">
                <span class="status-dot"></span>
                SYSTEM ONLINE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    '<div class="eyebrow">Scanner Settings</div>',
    unsafe_allow_html=True,
)

model_choice = st.sidebar.selectbox(
    "Detection model",
    [
        "YOLOv8s — Fast",
        "YOLO11 — High Accuracy",
    ],
)

confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05,
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    <div class="model-card">

        <div class="model-name">⚡ YOLOv8s</div>

        <div class="model-meta">
            SPEED OPTIMIZED<br>
            CarDD trained<br>
            Lightweight segmentation
        </div>

    </div>

    <div class="model-card">

        <div class="model-name">🎯 YOLO11</div>

        <div class="model-meta">
            ACCURACY FOCUSED<br>
            CarDD trained<br>
            Instance segmentation
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div style="margin-top:1rem;">
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.link_button(
    "🔗 YOLOv8 Model",
    "https://huggingface.co/abdullahg7/cardd-yolov8s",
)

st.sidebar.link_button(
    "🔗 YOLO11 Model",
    "https://huggingface.co/harpreetsahota/car-dd-segmentation-yolov11",
)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="eyebrow">01 — Vehicle Image</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown(
        """
        <div style="
            color:#8C949B;
            font-family:'IBM Plex Mono',monospace;
            font-size:0.68rem;
            margin-bottom:0.8rem;
        ">
            INPUT RECEIVED ✓
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:

        render_frame(
            image,
            "📷 Input Feed",
        )

    with col2:

        st.markdown(
            """
            <div class="frame-label">
                🤖 AI ANALYSIS
            </div>

            <div style="
                height:100%;
                min-height:100px;
                background:#171B1E;
                border:1px dashed rgba(237,234,225,0.10);
                display:flex;
                align-items:center;
                justify-content:center;
                color:#8C949B;
                font-family:'IBM Plex Mono',monospace;
                font-size:0.72rem;
                text-align:center;
                padding:2rem;
            ">
                READY FOR DIAGNOSTIC SCAN
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    run_scan = st.button(
        "🔍  RUN DIAGNOSTIC SCAN",
        use_container_width=True,
    )

    if run_scan:

        start_time = datetime.now()

        with st.spinner(
            f"Running {model_choice.split(' — ')[0]} diagnostic..."
        ):

            # Load selected model
            if model_choice == "YOLOv8s — Fast":
                model = load_yolov8()
            else:
                model = load_yolo11()

            # Run inference
            results = model.predict(
                source=np.array(image),
                conf=confidence,
                verbose=False,
            )

            # YOLO plot returns BGR
            plotted = results[0].plot()

            result_image = Image.fromarray(
                plotted[:, :, ::-1]
            )

        processing_time = (
            datetime.now() - start_time
        ).total_seconds()

        # ====================================================
        # RESULT IMAGE
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="eyebrow">02 — Detection Output</div>',
            unsafe_allow_html=True,
        )

        output_left, output_right = st.columns(
            2,
            gap="large",
        )

        with output_left:

            render_frame(
                result_image,
                "🎯 Annotated Output",
            )

        with output_right:

            st.markdown(
                '<div class="frame-label">📋 Scan Summary</div>',
                unsafe_allow_html=True,
            )

            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:

                detection_count = len(boxes)

                damages = get_unique_damage(
                    results,
                    model,
                )

                scores = [
                    float(box.conf[0])
                    for box in boxes
                ]

                average_confidence = (
                    sum(scores) / len(scores)
                )

                metric1, metric2 = st.columns(2)

                with metric1:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-icon">⚠️</div>
                            <div class="metric-value">
                                {detection_count}
                            </div>
                            <div class="metric-label">
                                Detections
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with metric2:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-icon">🎯</div>
                            <div class="metric-value">
                                {average_confidence:.0%}
                            </div>
                            <div class="metric-label">
                                Avg Confidence
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown(
                    f"""
                    <div class="info-card">

                        <div class="info-card-icon">
                            🧾
                        </div>

                        <div class="info-card-title">
                            Damage Categories
                        </div>

                        <div class="info-card-text">
                            {", ".join(damages)}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    """
                    <div class="clear-banner">

                        ✓ NO DAMAGE DETECTED

                        <br><br>

                        The selected model did not identify
                        damage above the configured confidence
                        threshold.

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ====================================================
        # DAMAGE REPORT
        # ====================================================

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            '<div class="eyebrow">03 — Detection Report</div>',
            unsafe_allow_html=True,
        )

        if (
            results[0].boxes is not None
            and len(results[0].boxes) > 0
        ):

            st.markdown(
                '<div class="report-title">🔧 Findings</div>',
                unsafe_allow_html=True,
            )

            for index, box in enumerate(results[0].boxes):

                class_id = int(box.cls[0])
                score = float(box.conf[0])

                damage_type = (
                    model.names[class_id]
                    .replace("_", " ")
                    .title()
                )

                confidence_label, confidence_class = (
                    confidence_level(score)
                )

                st.markdown(
                    f"""
                    <div class="ticket">

                        <span class="
                            confidence-badge
                            {confidence_class}
                        ">
                            {confidence_label}
                        </span>

                        <span class="name">
                            {damage_type}
                        </span>

                        <div class="bar-track">

                            <div
                                class="bar-fill"
                                style="width:{score * 100:.0f}%"
                            ></div>

                        </div>

                        <span class="score">
                            {score:.1%}
                        </span>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ====================================================
        # DOWNLOAD RESULT
        # ====================================================

        output_buffer = io.BytesIO()

        result_image.save(
            output_buffer,
            format="PNG",
        )

        st.download_button(
            label="⬇️ Download Annotated Image",
            data=output_buffer.getvalue(),
            file_name="cardd_damage_scan.png",
            mime="image/png",
            use_container_width=True,
        )

        # ====================================================
        # SCAN INFO
        # ====================================================

        st.markdown(
            f"""
            <div class="footer-strip">

                MODEL
                · {model_choice.split(" — ")[0]}

                &nbsp; · &nbsp;

                THRESHOLD
                · {confidence:.2f}

                &nbsp; · &nbsp;

                PROCESSING
                · {processing_time:.2f}s

                &nbsp; · &nbsp;

                ENGINE
                · CarDD Vision

                <br><br>

                ⚠️ AI confidence indicates model confidence,
                not physical damage severity.

            </div>
            """,
            unsafe_allow_html=True,
        )

else:

    # ========================================================
    # EMPTY STATE
    # ========================================================

    st.markdown(
        """
        <div style="
            margin-top:1rem;
            padding:3rem 2rem;
            text-align:center;
            background:#171B1E;
            border:1px dashed rgba(237,234,225,0.12);
            border-radius:5px;
        ">

            <div style="font-size:3rem;margin-bottom:0.7rem;">
                🚗
            </div>

            <div style="
                font-family:'Space Grotesk',sans-serif;
                font-size:1.2rem;
                font-weight:600;
            ">
                Vehicle scanner ready
            </div>

            <div style="
                margin-top:0.5rem;
                color:#8C949B;
                font-size:0.82rem;
            ">
                Upload a vehicle image to begin AI inspection.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-strip" style="text-align:center;">

        🚗 CarDD Vision

        &nbsp; · &nbsp;

        AI-assisted vehicle damage detection

        &nbsp; · &nbsp;

        Built with YOLO + CarDD

    </div>
    """,
    unsafe_allow_html=True,
)
```
