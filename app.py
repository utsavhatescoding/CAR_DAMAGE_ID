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
    --ink-2: #111C33;
    --blue: #2563FF;
    --blue-2: #4F46E5;
    --cyan: #06B6D4;
    --amber: #F59E0B;
    --surface: rgba(255,255,255,.96);
    --surface-2: #F7FAFF;
    --text: #0F172A;
    --muted: #64748B;
    --line: #DCE6F5;
    --soft-blue: #EDF4FF;
    --soft-cyan: #ECFEFF;
    --shadow: 0 18px 50px rgba(26, 54, 93, .10);
}

html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
    color: var(--text);
    background:
      radial-gradient(circle at 8% 4%, rgba(37,99,255,.13), transparent 28%),
      radial-gradient(circle at 92% 12%, rgba(6,182,212,.11), transparent 24%),
      linear-gradient(180deg, #F8FBFF 0%, #FFFFFF 42%, #F7FAFF 100%);
}

[data-testid="stHeader"] {
    background: rgba(248,251,255,.82);
    backdrop-filter: blur(18px);
    border-bottom: 1px solid rgba(220,230,245,.75);
}

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    visibility: hidden;
}

.block-container {
    max-width: 1160px;
    padding-top: 1.05rem;
    padding-bottom: 4.5rem;
}

h1, h2, h3, h4 {
    color: var(--ink);
    letter-spacing: -0.04em;
}

h1 {
    font-size: clamp(3rem, 6.4vw, 5.5rem) !important;
    line-height: .94 !important;
    font-weight: 850 !important;
    max-width: 900px;
}

h2 { font-weight: 820 !important; }
h3 { font-weight: 780 !important; }
p, .stCaption { color: var(--muted); }

/* hero + card surfaces */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(180deg, rgba(255,255,255,.99), rgba(250,252,255,.97));
    border: 1px solid rgba(194,211,236,.85) !important;
    border-radius: 24px !important;
    box-shadow: var(--shadow);
}

[data-testid="stImage"] img {
    border-radius: 20px;
    box-shadow: 0 12px 30px rgba(15,23,42,.08);
}

/* uploader — make it feel like the product */
[data-testid="stFileUploaderDropzone"] {
    background:
      linear-gradient(135deg, rgba(37,99,255,.065), rgba(6,182,212,.06)),
      #FFFFFF;
    border: 1.5px dashed #9BB8E8;
    border-radius: 22px;
    padding: 1.75rem;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,.7);
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--blue);
    background:
      linear-gradient(135deg, rgba(37,99,255,.10), rgba(6,182,212,.08)),
      #FFFFFF;
}
[data-testid="stFileUploaderDropzone"] button {
    border-radius: 12px !important;
    border: 1px solid #BFD1EF !important;
    background: white !important;
    color: var(--ink) !important;
    font-weight: 750 !important;
}

/* buttons */
.stButton > button, .stDownloadButton > button {
    min-height: 52px;
    border-radius: 14px;
    font-weight: 800;
    letter-spacing: -.01em;
    box-shadow: 0 8px 22px rgba(37,99,255,.14);
    transition: transform .16s ease, box-shadow .16s ease, filter .16s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(100deg, var(--blue), var(--blue-2) 58%, #6D5EF6);
    border: 0;
    color: white;
    box-shadow: 0 12px 28px rgba(37,99,255,.24);
}
.stButton > button[kind="primary"]:hover {
    filter: brightness(1.04);
    box-shadow: 0 15px 32px rgba(37,99,255,.30);
}

/* model selector */
[data-testid="stRadio"] > div { gap: .7rem; }
[data-testid="stRadio"] label {
    background: rgba(255,255,255,.94);
    border: 1px solid #D7E3F4;
    border-radius: 14px;
    padding: .78rem 1rem;
    box-shadow: 0 5px 16px rgba(26,54,93,.045);
    transition: all .16s ease;
}
[data-testid="stRadio"] label:hover {
    border-color: #9DBBEA;
    transform: translateY(-1px);
}
[data-testid="stRadio"] label:has(input:checked) {
    border-color: #7BA5FF;
    background: linear-gradient(135deg, #EEF4FF 0%, #ECFEFF 100%);
    box-shadow: 0 8px 22px rgba(37,99,255,.10);
}

[data-testid="stSlider"] [role="slider"] {
    background: linear-gradient(180deg, #2563FF, #4F46E5) !important;
    border: 2px solid white !important;
    box-shadow: 0 2px 8px rgba(37,99,255,.26);
}

/* metrics */
[data-testid="stMetric"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F7FAFF 100%);
    border: 1px solid #DCE6F5;
    border-radius: 18px;
    padding: 1rem 1.05rem;
    box-shadow: 0 10px 26px rgba(26,54,93,.065);
}
[data-testid="stMetricLabel"] { color: #718096; font-weight: 700; }
[data-testid="stMetricValue"] { color: var(--ink); font-weight: 850; }

/* tabs */
[data-baseweb="tab-list"] {
    gap: .75rem;
    border-bottom: 1px solid #DCE6F5;
}
[data-baseweb="tab"] {
    padding: .75rem .2rem .7rem;
    font-weight: 760;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: var(--blue);
}

[data-testid="stAlert"] {
    border-radius: 14px;
    border-width: 1px;
}
[data-testid="stNotificationContentInfo"] { color: var(--ink); }

[data-testid="stDataFrame"] {
    border: 1px solid #DCE6F5;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 20px rgba(26,54,93,.05);
}

[data-testid="stExpander"] {
    border: 1px solid #DCE6F5;
    border-radius: 14px;
    background: rgba(255,255,255,.86);
}

[data-testid="stSidebar"] {
    background:
      radial-gradient(circle at 20% 0%, rgba(37,99,255,.13), transparent 30%),
      #F9FBFF;
    border-right: 1px solid #DCE6F5;
}

hr { border-color: #DCE6F5 !important; }

/* native markdown micro-labels */
[data-testid="stCaptionContainer"] p {
    letter-spacing: .015em;
}

@media (max-width: 768px) {
    .block-container {
        padding: .72rem .84rem 3.25rem;
    }
    h1 {
        font-size: 3rem !important;
        line-height: .96 !important;
    }
    h2 { font-size: 1.62rem !important; }
    h3 { font-size: 1.12rem !important; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px !important;
        box-shadow: 0 12px 30px rgba(26,54,93,.08);
    }
    [data-testid="stFileUploaderDropzone"] {
        padding: 1.05rem;
        border-radius: 18px;
    }
    [data-testid="stRadio"] > div {
        flex-direction: column !important;
    }
    [data-testid="stRadio"] label {
        width: 100%;
        min-height: 50px;
        padding: .72rem .85rem;
    }
    .stButton > button, .stDownloadButton > button {
        width: 100%;
        min-height: 52px;
        border-radius: 14px;
    }
    [data-baseweb="tab-list"] {
        overflow-x: auto;
        white-space: nowrap;
        gap: 1rem;
    }
    [data-testid="column"] { min-width: 0 !important; }
    [data-testid="stImage"] img { border-radius: 16px; }
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
# SIDEBAR — secondary only
# ============================================================
with st.sidebar:
    if ICON_PATH.exists():
        st.image(str(ICON_PATH), width=52)
    st.markdown("### Nepal Vehicle Inspector")
    st.caption("AI vehicle damage assessment")
    st.divider()
    st.caption("MODEL OPTIONS")
    st.write("YOLO11m — Precision")
    st.write("YOLOv8s — Fast")
    st.divider()
    st.caption("Visual inspection only. Human review is recommended for safety, repair and insurance decisions.")

# ============================================================
# HEADER
# ============================================================
brand, nav = st.columns([4, 1], vertical_alignment="center")
with brand:
    a, b = st.columns([0.35, 4.5], vertical_alignment="center")
    with a:
        if ICON_PATH.exists():
            st.image(str(ICON_PATH), width=46)
    with b:
        st.markdown("### Nepal Vehicle Inspector")
        st.caption("AI-powered vehicle damage assessment")
with nav:
    st.caption("INSPECTION TOOL")

st.write("")
st.write("")

# ============================================================
# HERO — vibrant product-first hierarchy
# ============================================================
with st.container(border=True):
    st.caption("NEPAL VEHICLE INSPECTOR  ·  AI DAMAGE ASSESSMENT")
    st.title("See the damage.\nKnow what matters.")
    st.write(
        "Upload a vehicle photo and get an instant visual assessment with annotated damage, "
        "confidence scores and evidence for every detected area."
    )
    st.caption("⚡ Fast assessment   ·   🎯 Confidence-based findings   ·   🔎 Visual evidence")

st.write("")

# ============================================================
# ASSESSMENT
# ============================================================
st.subheader("Inspect a vehicle")
st.caption("Upload one clear exterior photo, choose the model, and run the assessment.")
st.write("")

model_choice = st.radio(
    "Detection model",
    ["YOLO11m — Precision", "YOLOv8s — Fast"],
    horizontal=True,
    help="YOLO11m is the recommended primary model. YOLOv8s is lighter and faster.",
)

with st.expander("Detection settings", expanded=False):
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Lower thresholds show more possible damage but can increase false positives.",
    )
    st.caption(f"Current threshold: {confidence:.0%}. Confidence is model certainty, not damage severity.")

uploaded_file = st.file_uploader(
    "Upload vehicle photo",
    type=["jpg", "jpeg", "png", "webp"],
    help="JPG, PNG or WEBP. Clear daylight images work best.",
)

if uploaded_file is None:
    st.info("Ready when you are — upload a vehicle image to begin the inspection.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    st.write("")
    preview, details = st.columns([1.55, 1], gap="large", vertical_alignment="center")
    with preview:
        st.image(image, use_container_width=True)
    with details:
        st.caption("READY TO ANALYZE")
        st.subheader("Vehicle photo loaded")
        st.write(f"**Model**  {model_choice}")
        st.write(f"**Threshold**  {confidence:.0%}")
        st.write(f"**Image**  {image.width} × {image.height}px")
        st.caption("The report will include the original image, annotated output, confidence scores and cropped evidence.")
        run = st.button("Analyze vehicle", type="primary", use_container_width=True)

    if run:
        try:
            if model_choice.startswith("YOLO11m"):
                with st.spinner("Analyzing vehicle with YOLO11m..."):
                    model = load_yolo11m()
                    output_image, detections, scan_time = run_scan(model, image, confidence)
                model_name = "YOLO11m"
            else:
                with st.spinner("Analyzing vehicle with YOLOv8s..."):
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
    st.divider()
    st.write("")

    st.caption(f"INSPECTION REPORT  ·  {inspection_id}")
    status_col, meta_col = st.columns([2, 1], vertical_alignment="bottom")
    with status_col:
        if detections:
            st.header(f"{len(detections)} visible damage finding{'s' if len(detections) != 1 else ''}")
            st.caption("Review the annotated image and every evidence region below.")
        else:
            st.header("No visible damage detected")
            st.caption("No detections were found above the selected confidence threshold.")
    with meta_col:
        st.caption(f"{result['model_name']} · {result['threshold']:.0%} threshold · {result['scan_time']:.2f}s")

    st.write("")
    m1, m2, m3 = st.columns(3)
    m1.metric("Findings", len(detections))
    m2.metric("Damage types", unique_types)
    m3.metric("Highest confidence", f"{highest:.0%}")

    st.write("")
    overview_tab, evidence_tab, detail_tab = st.tabs(["Overview", "Evidence", "Details"])

    with overview_tab:
        st.write("")
        before, after = st.columns(2, gap="large")
        with before:
            st.caption("ORIGINAL")
            st.image(result["original"], use_container_width=True)
        with after:
            st.caption("AI ASSESSMENT")
            st.image(result["annotated"], use_container_width=True)

        st.write("")
        st.subheader("Findings")
        if detections:
            summary_rows = []
            for i, d in enumerate(detections, start=1):
                x1, y1, x2, y2 = d["box"]
                summary_rows.append({
                    "Finding": f"F-{i:02d}",
                    "Damage": d["name"],
                    "Confidence": f"{d['confidence']:.1%}",
                    "Region": f"({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})",
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        else:
            st.success("No visible damage was detected above the selected threshold.")

    with evidence_tab:
        st.write("")
        if not detections:
            st.info("No evidence regions to review.")
        else:
            st.subheader("Damage evidence")
            st.caption("Each item shows the detected area and the model's confidence in that classification.")
            st.write("")
            for i, d in enumerate(detections, start=1):
                with st.container(border=True):
                    crop, info = st.columns([1, 1.35], gap="large", vertical_alignment="center")
                    with crop:
                        st.image(d["crop"], use_container_width=True)
                    with info:
                        st.caption(f"FINDING F-{i:02d}")
                        st.subheader(d["name"])
                        st.metric("Confidence", f"{d['confidence']:.1%}")
                        st.progress(min(max(d["confidence"], 0.0), 1.0))
                        x1, y1, x2, y2 = d["box"]
                        st.caption(f"Detected region: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})")
                        st.caption("Confidence reflects model certainty; it is not a repair-cost or severity score.")

    with detail_tab:
        st.write("")
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Assessment")
            st.write(f"**Inspection ID:** {inspection_id}")
            st.write(f"**Timestamp:** {result['timestamp']}")
            st.write(f"**Source file:** {st.session_state.inspection_source_name}")
            st.write(f"**Image size:** {result['original'].width} × {result['original'].height}px")
        with right:
            st.subheader("Model")
            st.write(f"**Model:** {result['model_name']}")
            st.write(f"**Threshold:** {result['threshold']:.0%}")
            st.write(f"**Inference time:** {result['scan_time']:.3f}s")
            st.write(f"**Detected regions:** {len(detections)}")

        if detections:
            export_rows = []
            for i, d in enumerate(detections, start=1):
                x1, y1, x2, y2 = d["box"]
                export_rows.append({
                    "inspection_id": inspection_id,
                    "finding_id": f"F-{i:02d}",
                    "damage": d["name"],
                    "confidence": round(d["confidence"], 4),
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                })
            csv_bytes = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download findings CSV",
                data=csv_bytes,
                file_name=f"{inspection_id}_findings.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.write("")
    st.caption("AI-assisted visual screening only. A qualified human inspector should review findings before safety, valuation, repair or insurance decisions.")

# ============================================================
# FOOTER
# ============================================================
st.write("")
st.write("")
st.divider()
left, right = st.columns([3, 1])
with left:
    st.caption("Nepal Vehicle Inspector · AI-powered vehicle damage assessment")
with right:
    st.caption("NVI · 2026")
