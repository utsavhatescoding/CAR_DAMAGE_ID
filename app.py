import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image
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
# PRODUCT THEME
# Visible UI uses native Streamlit components only.
# CSS below changes styling; it does not render visible HTML content.
# ============================================================
st.markdown(
    """
<style>
:root {
    --ink: #0B1220;
    --navy: #10213B;
    --navy-2: #16345B;
    --steel: #35546F;
    --paper: #F4F7FA;
    --surface: #FFFFFF;
    --line: #D9E2EA;
    --muted: #687786;
    --signal: #F2B84B;
    --signal-soft: #FFF5DA;
    --good: #167A58;
    --danger: #B83A3A;
}

html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 88% 0%, rgba(22,52,91,.08), transparent 30rem),
        var(--paper);
    color: var(--ink);
}

[data-testid="stHeader"] {
    background: rgba(244,247,250,.88);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(217,226,234,.72);
}

#MainMenu,
footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    visibility: hidden;
}

.block-container {
    max-width: 1180px;
    padding-top: 1.4rem;
    padding-bottom: 4rem;
}

/* brand image */
[data-testid="stImage"] img {
    border-radius: 18px;
}

/* headings */
h1, h2, h3, h4 {
    letter-spacing: -.025em;
    color: var(--ink);
}

h1 {
    font-size: clamp(2rem, 5vw, 3.65rem) !important;
    line-height: 1.02 !important;
    font-weight: 800 !important;
}

h2 { font-weight: 780 !important; }
h3 { font-weight: 750 !important; }

p, .stCaption { color: var(--muted); }

/* cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,.98);
    border: 1px solid var(--line) !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 34px rgba(13,32,51,.045);
}

/* file uploader */
[data-testid="stFileUploaderDropzone"] {
    background:
        linear-gradient(180deg, #FFFFFF, #FAFCFE);
    border: 1.6px dashed #9DB0C1;
    border-radius: 18px;
    padding: 1.45rem;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--navy-2);
    background: #F9FBFD;
}

/* primary button */
.stButton > button {
    min-height: 48px;
    border-radius: 13px;
    font-weight: 750;
    letter-spacing: -.01em;
}

.stButton > button[kind="primary"] {
    background: var(--navy);
    border-color: var(--navy);
    color: #fff;
    box-shadow: 0 8px 20px rgba(16,33,59,.15);
}

.stButton > button[kind="primary"]:hover {
    background: var(--navy-2);
    border-color: var(--navy-2);
}

/* radio controls */
[data-testid="stRadio"] > div {
    gap: .55rem;
}
[data-testid="stRadio"] label {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: .58rem .82rem;
    min-height: 42px;
}
[data-testid="stRadio"] label:has(input:checked) {
    border-color: var(--navy-2);
    background: #EEF4FA;
}

/* slider accent */
[data-testid="stSlider"] [role="slider"] {
    background: var(--navy-2) !important;
}

/* metrics */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1rem 1.05rem;
    box-shadow: 0 6px 22px rgba(13,32,51,.035);
}
[data-testid="stMetricLabel"] { color: var(--muted); }
[data-testid="stMetricValue"] { color: var(--ink); font-weight: 800; }

/* tabs */
[data-baseweb="tab-list"] {
    gap: .45rem;
    background: #EAF0F5;
    border-radius: 13px;
    padding: .3rem;
}
[data-baseweb="tab"] {
    border-radius: 10px;
    padding: .55rem .9rem;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #fff;
    box-shadow: 0 2px 7px rgba(13,32,51,.08);
}

/* alerts */
[data-testid="stAlert"] {
    border-radius: 14px;
}

/* dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 16px;
    overflow: hidden;
}

/* desktop advanced panel */
[data-testid="stSidebar"] {
    background: #F8FAFC;
    border-right: 1px solid var(--line);
}

/* mobile */
@media (max-width: 768px) {
    .block-container {
        padding: .8rem .85rem 3rem .85rem;
    }

    h1 {
        font-size: 2.2rem !important;
        line-height: 1.05 !important;
    }

    h2 { font-size: 1.55rem !important; }
    h3 { font-size: 1.14rem !important; }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        padding: .9rem;
        border-radius: 14px;
    }

    [data-testid="stMetric"] {
        padding: .78rem .85rem;
        border-radius: 13px;
    }

    .stButton > button {
        min-height: 50px;
        width: 100%;
    }

    [data-testid="stRadio"] > div {
        flex-direction: column !important;
    }

    [data-testid="stRadio"] label {
        width: 100%;
    }

    [data-baseweb="tab-list"] {
        overflow-x: auto;
        white-space: nowrap;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# MODELS
# ============================================================
@st.cache_resource(show_spinner=False)
def load_yolo11m():
    model_dir = Path.home() / ".cache" / "cardd_vision"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "yolo11m_car_damage.pt"
    if not model_path.exists():
        urllib.request.urlretrieve(
            "https://github.com/ReverendBayes/YOLO11m-Car-Damage-Detector/raw/main/trained.pt",
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
    results = model.predict(source=np.array(image), conf=confidence, verbose=False)
    scan_time = time.perf_counter() - start_time
    result = results[0]

    plotted = result.plot()
    output_image = Image.fromarray(plotted[:, :, ::-1])

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            score = float(box.conf[0])
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            damage_name = clean_damage_name(model.names[class_id])
            detections.append(
                {
                    "name": damage_name,
                    "confidence": score,
                    "box": xyxy,
                    "crop": get_damage_crop(image, xyxy),
                }
            )

    detections.sort(key=lambda x: x["confidence"], reverse=True)
    return output_image, detections, scan_time


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
# ADVANCED SIDEBAR — optional on desktop/mobile
# Model selection is ALSO in the main workflow so mobile users never
# need the sidebar.
# ============================================================
with st.sidebar:
    if ICON_PATH.exists():
        st.image(str(ICON_PATH), width=58)
    st.markdown("### Nepal Vehicle Inspector")
    st.caption("Field inspection console")
    st.divider()
    st.markdown("**About this inspection**")
    st.caption(
        "AI-assisted visible vehicle damage screening. Use findings as inspection evidence, "
        "not as a substitute for mechanical or safety inspection."
    )
    st.divider()
    st.markdown("**Models available**")
    st.write("YOLO11m — Precision")
    st.write("YOLOv8s — Fast")

# ============================================================
# PRODUCT HEADER
# ============================================================
brand_col, status_col = st.columns([5, 1.2], vertical_alignment="center")
with brand_col:
    icon_col, name_col = st.columns([0.45, 4.8], vertical_alignment="center")
    with icon_col:
        if ICON_PATH.exists():
            st.image(str(ICON_PATH), width=66)
        else:
            st.markdown("## 🛡️")
    with name_col:
        st.markdown("### Nepal Vehicle Inspector")
        st.caption("AI-assisted exterior vehicle inspection · Build 2026.08.22b")
with status_col:
    st.success("SYSTEM READY")

st.write("")

hero_left, hero_right = st.columns([1.45, 1], gap="large", vertical_alignment="center")
with hero_left:
    st.caption("FIELD INSPECTION · VEHICLE DAMAGE INTELLIGENCE")
    st.title("Inspect the vehicle.\nDocument the evidence.")
    st.write(
        "A field-ready visual inspection tool for identifying visible exterior damage, "
        "reviewing model confidence and documenting every detected region."
    )
with hero_right:
    with st.container(border=True):
        st.markdown("#### Inspector protocol")
        st.write("**01**  Capture a clear vehicle image")
        st.write("**02**  Select the inspection model")
        st.write("**03**  Review annotated evidence")
        st.write("**04**  Check every confidence score")
        st.caption("Visual screening only · Human review required")

st.write("")

# ============================================================
# NEW INSPECTION
# ============================================================
with st.container(border=True):
    top_left, top_right = st.columns([1.8, 1], vertical_alignment="bottom")
    with top_left:
        st.subheader("New vehicle inspection")
        st.caption("Upload one exterior vehicle image. JPG, PNG or WEBP.")
    with top_right:
        st.caption("Inspection desk · Nepal")

    st.divider()

    # IMPORTANT: model control lives in main UI for mobile access.
    st.markdown("#### 1. Select inspection model")
    model_choice = st.radio(
        "Inspection model",
        ["YOLO11m — Precision", "YOLOv8s — Fast"],
        horizontal=True,
        label_visibility="collapsed",
        help="YOLO11m is recommended for the primary inspection. YOLOv8s is lighter and faster.",
    )

    sensitivity_col, note_col = st.columns([1.4, 1], vertical_alignment="center")
    with sensitivity_col:
        confidence = st.slider(
            "Detection threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.25,
            step=0.05,
            help="Lower thresholds show more possible damage but can increase false positives.",
        )
    with note_col:
        st.caption(
            f"Current threshold: **{confidence:.0%}** · "
            "Confidence indicates model certainty, not damage severity."
        )

    st.write("")
    st.markdown("#### 2. Upload vehicle image")
    uploaded_file = st.file_uploader(
        "Vehicle image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        help="For best results, use a clear, well-lit image with the damaged area visible.",
    )

    if uploaded_file is None:
        st.info(
            "Capture or upload a clear vehicle image to start the inspection."
        )
    else:
        image = Image.open(uploaded_file).convert("RGB")
        preview_col, action_col = st.columns([1.6, 1], gap="large", vertical_alignment="center")
        with preview_col:
            st.image(image, caption="Vehicle submitted for inspection", use_container_width=True)
        with action_col:
            st.markdown("#### Ready for inspection")
            st.write(f"**Model:** {model_choice}")
            st.write(f"**Threshold:** {confidence:.0%}")
            st.write(f"**Image:** {image.width} × {image.height}px")
            st.caption(
                "The output will include the annotated vehicle, all detections, "
                "confidence values, evidence crops and technical details."
            )
            run = st.button(
                "Run AI inspection",
                type="primary",
                use_container_width=True,
            )

        if run:
            try:
                if model_choice.startswith("YOLO11m"):
                    with st.spinner("Inspector is analysing the vehicle with YOLO11m..."):
                        model = load_yolo11m()
                        output_image, detections, scan_time = run_scan(model, image, confidence)
                    model_name = "YOLO11m"
                else:
                    with st.spinner("Inspector is analysing the vehicle with YOLOv8s..."):
                        model = load_yolov8()
                        output_image, detections, scan_time = run_scan(model, image, confidence)
                    model_name = "YOLOv8s"

                st.session_state.inspection_result = {
                    "original": image.copy(),
                    "annotated": output_image,
                    "detections": detections,
                    "scan_time": scan_time,
                    "model_name": model_name,
                    "threshold": confidence,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                st.session_state.inspection_source_name = uploaded_file.name
                st.session_state.inspection_id = datetime.now().strftime("NVI-%y%m%d-%H%M%S")
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

    report_title, report_status = st.columns([3, 1], vertical_alignment="center")
    with report_title:
        st.caption(f"INSPECTION RECORD · {inspection_id}")
        st.header("Vehicle inspection report")
        st.caption(
            f"{st.session_state.inspection_source_name or 'Vehicle image'} · "
            f"{result['timestamp']} · {result['model_name']} · threshold {result['threshold']:.0%}"
        )
    with report_status:
        if detections:
            st.warning(f"{len(detections)} finding(s)")
        else:
            st.success("No findings")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Detected regions", len(detections))
    m2.metric("Damage classes", unique_types)
    m3.metric("Peak confidence", f"{highest:.0%}")
    m4.metric("Inference", f"{result['scan_time']:.2f}s")

    st.write("")

    overview_tab, evidence_tab, detail_tab = st.tabs(
        ["Inspection overview", "Evidence", "Technical record"]
    )

    with overview_tab:
        st.markdown("### Original vs AI inspection")
        before, after = st.columns(2, gap="large")
        with before:
            st.caption("ORIGINAL VEHICLE")
            st.image(result["original"], use_container_width=True)
        with after:
            st.caption("AI INSPECTION OUTPUT")
            st.image(result["annotated"], use_container_width=True)

        st.write("")
        st.markdown("### Findings summary")
        if detections:
            summary_rows = []
            for i, d in enumerate(detections, start=1):
                x1, y1, x2, y2 = d["box"]
                summary_rows.append(
                    {
                        "ID": f"F-{i:02d}",
                        "Damage": d["name"],
                        "Confidence": f"{d['confidence']:.1%}",
                        "Evidence box": f"({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})",
                    }
                )
            st.dataframe(
                pd.DataFrame(summary_rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success(
                "No visible damage was detected above the selected threshold."
            )

    with evidence_tab:
        if not detections:
            st.success("No evidence regions to review.")
        else:
            st.markdown("### Detection evidence")
            st.caption(
                "Every finding below shows the exact crop used as visual evidence and the model confidence."
            )
            for i, d in enumerate(detections, start=1):
                with st.container(border=True):
                    evidence_col, info_col = st.columns([1.05, 1.35], gap="large", vertical_alignment="center")
                    with evidence_col:
                        st.image(d["crop"], caption=f"Evidence F-{i:02d}", use_container_width=True)
                    with info_col:
                        st.caption(f"FINDING F-{i:02d}")
                        st.subheader(d["name"])
                        st.metric("Model confidence", f"{d['confidence']:.1%}")
                        st.progress(min(max(d["confidence"], 0.0), 1.0))
                        x1, y1, x2, y2 = d["box"]
                        st.write(f"**Detected region:** ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})")
                        st.caption(
                            "Confidence is the model's certainty about the classification. "
                            "It does not measure repair cost, physical severity or roadworthiness."
                        )

    with detail_tab:
        tech_left, tech_right = st.columns(2)
        with tech_left:
            with st.container(border=True):
                st.markdown("#### Inspection record")
                st.write(f"**Inspection ID:** {inspection_id}")
                st.write(f"**Timestamp:** {result['timestamp']}")
                st.write(f"**Source file:** {st.session_state.inspection_source_name}")
                st.write(f"**Image size:** {result['original'].width} × {result['original'].height}px")
        with tech_right:
            with st.container(border=True):
                st.markdown("#### AI record")
                st.write(f"**Model:** {result['model_name']}")
                st.write(f"**Threshold:** {result['threshold']:.0%}")
                st.write(f"**Inference time:** {result['scan_time']:.3f}s")
                st.write(f"**Detected regions:** {len(detections)}")

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
            csv_bytes = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download findings CSV",
                data=csv_bytes,
                file_name=f"{inspection_id}_findings.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.write("")
    st.warning(
        "AI-assisted visual screening only. A qualified human inspector should review findings "
        "before safety, valuation, repair or insurance decisions."
    )

# ============================================================
# FOOTER
# ============================================================
st.write("")
st.divider()
footer_left, footer_right = st.columns([3, 1])
with footer_left:
    st.caption("Nepal Vehicle Inspector · AI-assisted vehicle damage screening")
with footer_right:
    st.caption("Field Console v1")
