import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CarDD Vision",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&
        family=Manrope:wght@500;600;700;800&display=swap'
    );

    :root {
        --bg: #F6F7F9;
        --surface: #FFFFFF;
        --surface-soft: #F0F2F5;
        --text: #101828;
        --muted: #667085;
        --border: #E4E7EC;
        --dark: #111827;
        --blue: #2563EB;
        --blue-soft: #EFF6FF;
        --orange: #F97316;
        --orange-soft: #FFF7ED;
        --green: #16A34A;
        --green-soft: #F0FDF4;
        --red: #DC2626;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    #MainMenu,
    footer {
        visibility: hidden;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* SIDEBAR */

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }


    /* BRAND */

    .brand {
        display: flex;
        align-items: center;
        gap: 11px;
        margin-bottom: 2rem;
    }

    .brand-icon {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: var(--dark);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }

    .brand-title {
        font-family: 'Manrope', sans-serif;
        font-size: 1.05rem;
        font-weight: 800;
        color: var(--dark);
        line-height: 1.1;
    }

    .brand-sub {
        color: var(--muted);
        font-size: 0.72rem;
        margin-top: 2px;
    }


    /* HERO */

    .hero {
        background:
            radial-gradient(
                circle at 85% 20%,
                rgba(37,99,235,0.16),
                transparent 28%
            ),
            var(--dark);

        color: white;
        padding: 3.2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        border: 1px solid rgba(255,255,255,0.08);
        right: -80px;
        bottom: -130px;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: #93C5FD;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .eyebrow-dot {
        width: 8px;
        height: 8px;
        background: #60A5FA;
        border-radius: 50%;
        box-shadow: 0 0 12px #60A5FA;
    }

    .hero h1 {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2.4rem, 5vw, 4.2rem);
        font-weight: 800;
        line-height: 1.05;
        margin: 0;
        max-width: 720px;
        letter-spacing: -0.04em;
    }

    .hero h1 span {
        color: #60A5FA;
    }

    .hero p {
        color: #B8C1D1;
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 620px;
        margin-top: 1.2rem;
        margin-bottom: 0;
    }


    /* SECTION */

    .section-label {
        font-size: 0.78rem;
        font-weight: 800;
        color: var(--muted);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }


    /* CARDS */

    .info-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.3rem;
        height: 100%;
    }

    .info-icon {
        font-size: 1.45rem;
        margin-bottom: 0.7rem;
    }

    .info-title {
        font-family: 'Manrope', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.35rem;
        color: var(--text);
    }

    .info-text {
        color: var(--muted);
        font-size: 0.84rem;
        line-height: 1.55;
    }


    /* RESULT HEADER */

    .result-hero {
        background: white;
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.7rem;
        margin-bottom: 1.5rem;
    }

    .result-title {
        font-family: 'Manrope', sans-serif;
        font-size: 1.65rem;
        font-weight: 800;
        margin: 0;
        color: var(--dark);
    }

    .result-sub {
        color: var(--muted);
        margin-top: 0.35rem;
        font-size: 0.9rem;
    }


    /* METRICS */

    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.25rem;
        height: 100%;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .metric-value {
        font-family: 'Manrope', sans-serif;
        color: var(--dark);
        font-size: 1.7rem;
        font-weight: 800;
        margin-top: 0.4rem;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.76rem;
        margin-top: 0.2rem;
    }


    /* FINDING CARDS */

    .finding {
        background: white;
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
    }

    .finding-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
    }

    .finding-name {
        font-family: 'Manrope', sans-serif;
        font-size: 1rem;
        font-weight: 800;
        color: var(--dark);
    }

    .confidence-pill {
        background: var(--blue-soft);
        color: var(--blue);
        border-radius: 999px;
        padding: 0.32rem 0.65rem;
        font-size: 0.75rem;
        font-weight: 800;
        white-space: nowrap;
    }

    .progress-track {
        height: 7px;
        background: #EEF0F3;
        border-radius: 99px;
        overflow: hidden;
        margin-top: 0.8rem;
    }

    .progress-fill {
        height: 100%;
        background: linear-gradient(
            90deg,
            #2563EB,
            #60A5FA
        );
        border-radius: 99px;
    }

    .finding-note {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 0.65rem;
    }


    /* IMAGE CONTAINER */

    .image-label {
        font-family: 'Manrope', sans-serif;
        font-size: 1rem;
        font-weight: 800;
        color: var(--dark);
        margin-bottom: 0.6rem;
    }


    /* STATUS */

    .status-good {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: var(--green-soft);
        color: var(--green);
        padding: 0.42rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 800;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--green);
    }


    /* UPLOADER */

    [data-testid="stFileUploaderDropzone"] {
        background: white;
        border: 2px dashed #CBD5E1;
        border-radius: 20px;
        padding: 2rem;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--blue);
        background: #FAFCFF;
    }


    /* BUTTON */

    .stButton > button {
        width: 100%;
        background: var(--dark);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        font-family: 'Manrope', sans-serif;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: var(--blue);
        border: none;
        transform: translateY(-1px);
    }


    /* IMAGE */

    [data-testid="stImage"] img {
        border-radius: 16px;
    }


    /* MOBILE */

    @media (max-width: 768px) {

        .hero {
            padding: 2rem 1.4rem;
            border-radius: 18px;
        }

        .hero h1 {
            font-size: 2.4rem;
        }

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADERS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_yolo11m():

    model_dir = Path.home() / ".cache" / "cardd_vision"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "yolo11m_car_damage.pt"

    if not model_path.exists():

        urllib.request.urlretrieve(
            "https://github.com/ReverendBayes/"
            "YOLO11m-Car-Damage-Detector/"
            "raw/main/trained.pt",
            str(model_path),
        )

    return YOLO(str(model_path))


@st.cache_resource(show_spinner=False)
def load_yolov8():

    model_path = hf_hub_download(
        repo_id="abdullahg7/cardd-yolov8s",
        filename="v2.0/best.pt",
    )

    return YOLO(model_path)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

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


def run_scan(model, image, confidence):

    start_time = time.perf_counter()

    results = model.predict(
        source=np.array(image),
        conf=confidence,
        verbose=False,
    )

    scan_time = time.perf_counter() - start_time

    result = results[0]

    plotted = result.plot()

    output_image = Image.fromarray(
        plotted[:, :, ::-1]
    )

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            score = float(box.conf[0])

            damage_name = clean_damage_name(
                model.names[class_id]
            )

            xyxy = box.xyxy[0].tolist()

            crop = get_damage_crop(
                image,
                xyxy
            )

            detections.append(
                {
                    "name": damage_name,
                    "confidence": score,
                    "crop": crop,
                    "box": xyxy,
                }
            )

    detections = sorted(
        detections,
        key=lambda x: x["confidence"],
        reverse=True,
    )

    return output_image, detections, scan_time


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🚘</div>

            <div>
                <div class="brand-title">
                    CarDD Vision
                </div>

                <div class="brand-sub">
                    AI VEHICLE INSPECTION
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">Inspection mode</div>',
        unsafe_allow_html=True,
    )

    model_choice = st.radio(
        "Model",
        [
            "🔬 YOLO11m — Recommended",
            "⚡ YOLOv8s — Fast Scan",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-label">Detection sensitivity</div>',
        unsafe_allow_html=True,
    )

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        label_visibility="collapsed",
    )

    st.caption(
        f"Current threshold: {confidence:.0%}"
    )

    st.markdown("---")

    if "YOLO11m" in model_choice:

        st.markdown(
            """
            <div class="info-card">

                <div class="info-icon">🔬</div>

                <div class="info-title">
                    Recommended
                </div>

                <div class="info-text">
                    Balanced detection model for
                    dents, scratches, cracks,
                    broken lamps, shattered glass
                    and flat tires.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="info-card">

                <div class="info-icon">⚡</div>

                <div class="info-title">
                    Fast Scan
                </div>

                <div class="info-text">
                    Lightweight segmentation model
                    for quicker vehicle inspection.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="eyebrow">
            <span class="eyebrow-dot"></span>
            AI-powered vehicle inspection
        </div>

        <h1>
            See the damage.<br>
            <span>Understand the vehicle.</span>
        </h1>

        <p>
            Upload a vehicle photo and let CarDD Vision
            identify visible damage in seconds.
            Review the inspection, detected areas and
            confidence for every finding.
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HOW IT WORKS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">📸</div>

            <div class="info-title">
                01 — Upload
            </div>

            <div class="info-text">
                Add a clear photo of the vehicle.
                Exterior damage works best when
                visible and well lit.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">🧠</div>

            <div class="info-title">
                02 — AI Inspection
            </div>

            <div class="info-text">
                The selected vision model scans
                the image for visible damage
                patterns.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">🔍</div>

            <div class="info-title">
                03 — Review
            </div>

            <div class="info-text">
                Compare the original image,
                inspection output and individual
                damage evidence.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown("<br><br>", unsafe_allow_html=True)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="section-label">Start an inspection</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload vehicle image",
    type=["jpg", "jpeg", "png", "webp"],
)


# ============================================================
# BEFORE SCAN
# ============================================================

if uploaded_file is None:

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-card"
             style="text-align:center;padding:2.5rem;">

            <div style="font-size:2.3rem;margin-bottom:0.8rem;">
                🚗
            </div>

            <div class="info-title"
                 style="font-size:1.15rem;">
                Ready for inspection
            </div>

            <div class="info-text"
                 style="max-width:460px;margin:0.6rem auto 0;">
                Upload a clear image of a vehicle to begin
                the AI-powered damage inspection.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# IMAGE LOADED
# ============================================================

else:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("<br>", unsafe_allow_html=True)

    preview_col1, preview_col2 = st.columns([1.45, 1])

    with preview_col1:

        st.markdown(
            '<div class="image-label">Vehicle photo</div>',
            unsafe_allow_html=True,
        )

        st.image(
            image,
            use_container_width=True,
        )

    with preview_col2:

        st.markdown(
            """
            <div class="info-card"
                 style="height:auto;">

                <div class="info-icon">
                    🚘
                </div>

                <div class="info-title"
                     style="font-size:1.15rem;">
                    Ready to inspect
                </div>

                <div class="info-text"
                     style="margin-top:0.6rem;">

                    CarDD Vision will scan this image
                    for visible dents, scratches,
                    cracks, broken lamps,
                    shattered glass and flat tires.

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        scan_button = st.button(
            "🔍 Start AI Inspection",
            use_container_width=True,
        )


    # ========================================================
    # RUN SCAN
    # ========================================================

    if scan_button:

        if "YOLO11m" in model_choice:

            model_name = "YOLO11m"
            loading_message = "Loading YOLO11m and inspecting vehicle..."

            with st.spinner(loading_message):

                model = load_yolo11m()

                output_image, detections, scan_time = run_scan(
                    model,
                    image,
                    confidence,
                )

        else:

            model_name = "YOLOv8s"

            with st.spinner(
                "Running fast vehicle inspection..."
            ):

                model = load_yolov8()

                output_image, detections, scan_time = run_scan(
                    model,
                    image,
                    confidence,
                )


        # ====================================================
        # RESULT HEADER
        # ====================================================

        st.markdown("<br><br>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="result-hero">

                <div class="status-good">
                    <span class="status-dot"></span>
                    INSPECTION COMPLETE
                </div>

                <div class="result-title"
                     style="margin-top:1rem;">
                    Vehicle inspection report
                </div>

                <div class="result-sub">
                    {model_name} analysed the uploaded image
                    using a {confidence:.0%} confidence threshold.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        # ====================================================
        # METRICS
        # ====================================================

        highest_confidence = (
            max(
                [d["confidence"] for d in detections],
                default=0
            )
        )

        unique_damage_types = len(
            set(
                d["name"]
                for d in detections
            )
        )

        metric1, metric2, metric3, metric4 = st.columns(4)

        with metric1:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Findings
                    </div>

                    <div class="metric-value">
                        {len(detections)}
                    </div>

                    <div class="metric-note">
                        Total detected regions
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with metric2:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Damage types
                    </div>

                    <div class="metric-value">
                        {unique_damage_types}
                    </div>

                    <div class="metric-note">
                        Unique categories detected
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with metric3:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Highest confidence
                    </div>

                    <div class="metric-value">
                        {highest_confidence:.0%}
                    </div>

                    <div class="metric-note">
                        Strongest model prediction
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with metric4:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Scan time
                    </div>

                    <div class="metric-value">
                        {scan_time:.2f}s
                    </div>

                    <div class="metric-note">
                        Model inference time
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


        # ====================================================
        # BEFORE / AFTER
        # ====================================================

        st.markdown("<br><br>", unsafe_allow_html=True)

        before_col, after_col = st.columns(2)

        with before_col:

            st.markdown(
                '<div class="image-label">📷 Original vehicle</div>',
                unsafe_allow_html=True,
            )

            st.image(
                image,
                use_container_width=True,
            )

        with after_col:

            st.markdown(
                '<div class="image-label">✨ AI inspection output</div>',
                unsafe_allow_html=True,
            )

            st.image(
                output_image,
                use_container_width=True,
            )


        # ====================================================
        # DAMAGE FINDINGS
        # ====================================================

        st.markdown("<br><br>", unsafe_allow_html=True)

        st.markdown(
            '<div class="section-label">Detected damage evidence</div>',
            unsafe_allow_html=True,
        )


        if detections:

            st.markdown(
                """
                <div style="
                    font-family:'Manrope',sans-serif;
                    font-size:1.55rem;
                    font-weight:800;
                    margin-bottom:0.4rem;
                ">
                    Review each finding
                </div>

                <div style="
                    color:#667085;
                    font-size:0.9rem;
                    margin-bottom:1.4rem;
                ">
                    Each card shows the exact image region
                    identified by the model.
                </div>
                """,
                unsafe_allow_html=True,
            )


            for index, detection in enumerate(
                detections,
                start=1,
            ):

                finding_col, crop_col = st.columns(
                    [1.3, 1]
                )

                with finding_col:

                    st.markdown(
                        f"""
                        <div class="finding">

                            <div class="finding-header">

                                <div>

                                    <div class="finding-name">
                                        {index:02d}. {detection["name"]}
                                    </div>

                                </div>

                                <div class="confidence-pill">
                                    {detection["confidence"]:.0%} confidence
                                </div>

                            </div>

                            <div class="progress-track">

                                <div
                                    class="progress-fill"
                                    style="
                                        width:
                                        {detection["confidence"] * 100:.0f}%;
                                    ">
                                </div>

                            </div>

                            <div class="finding-note">
                                AI detected this region as
                                <b>{detection["name"].lower()}</b>.
                                Confidence represents the model's
                                certainty in the classification,
                                not the physical severity or repair cost.
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with crop_col:

                    st.image(
                        detection["crop"],
                        caption=f"Detected region {index}",
                        use_container_width=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)


        else:

            st.markdown(
                """
                <div class="info-card"
                     style="
                        text-align:center;
                        padding:2.5rem;
                        border-color:#BBF7D0;
                     ">

                    <div style="font-size:2.5rem;">
                        ✅
                    </div>

                    <div class="info-title"
                         style="
                            font-size:1.2rem;
                            margin-top:0.7rem;
                         ">
                        No visible damage detected
                    </div>

                    <div class="info-text"
                         style="
                            max-width:520px;
                            margin:0.6rem auto;
                         ">

                        The selected model did not identify
                        any damage above the selected confidence
                        threshold.

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.markdown("<br><br>", unsafe_allow_html=True)

        st.caption(
            "CarDD Vision is an AI-assisted visual inspection tool. "
            "Results should be reviewed by a qualified human inspector. "
            "Detection confidence is not a measure of damage severity, "
            "repair cost or vehicle safety."
        )
