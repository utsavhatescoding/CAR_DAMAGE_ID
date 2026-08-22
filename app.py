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
    --bg: #FBFCFE;
    --surface: #FFFFFF;
    --text: #111827;
    --muted: #667085;
    --line: #E7EAF0;
    --blue: #246BFD;
    --blue-2: #1859DB;
    --blue-soft: #EEF4FF;
    --green: #159467;
    --amber: #B7791F;
}

html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(251,252,254,.92);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(231,234,240,.78);
}

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    visibility: hidden;
}

.block-container {
    max-width: 1120px;
    padding-top: 1.1rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4 {
    color: var(--text);
    letter-spacing: -0.035em;
}

h1 {
    font-size: clamp(2.55rem, 6vw, 4.9rem) !important;
    line-height: .98 !important;
    font-weight: 780 !important;
    max-width: 820px;
}

h2 { font-weight: 740 !important; }
h3 { font-weight: 720 !important; }
p, .stCaption { color: var(--muted); }

[data-testid="stImage"] img {
    border-radius: 18px;
}

/* minimal card */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border: 1px solid var(--line) !important;
    border-radius: 18px !important;
    box-shadow: 0 10px 28px rgba(16,24,40,.045);
}

/* upload */
[data-testid="stFileUploaderDropzone"] {
    background: #FFFFFF;
    border: 1.5px dashed #C7CFDA;
    border-radius: 16px;
    padding: 1.5rem;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--blue);
    background: #FAFCFF;
}

/* buttons */
.stButton > button, .stDownloadButton > button {
    min-height: 48px;
    border-radius: 11px;
    font-weight: 700;
    box-shadow: none;
}
.stButton > button[kind="primary"] {
    background: var(--blue);
    border-color: var(--blue);
    color: #fff;
}
.stButton > button[kind="primary"]:hover {
    background: var(--blue-2);
    border-color: var(--blue-2);
}

/* radio as clean segmented cards */
[data-testid="stRadio"] > div { gap: .55rem; }
[data-testid="stRadio"] label {
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 11px;
    padding: .62rem .9rem;
}
[data-testid="stRadio"] label:has(input:checked) {
    border-color: #AFC7FF;
    background: var(--blue-soft);
}

[data-testid="stSlider"] [role="slider"] { background: var(--blue) !important; }

/* metrics */
[data-testid="stMetric"] {
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: .9rem 1rem;
}
[data-testid="stMetricValue"] { color: var(--text); font-weight: 760; }

/* tabs */
[data-baseweb="tab-list"] {
    gap: 1.25rem;
    border-bottom: 1px solid var(--line);
}
[data-baseweb="tab"] { padding: .65rem 0; }
[aria-selected="true"][data-baseweb="tab"] { color: var(--blue); }

[data-testid="stAlert"] {
    border-radius: 12px;
    border-width: 1px;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 14px;
    overflow: hidden;
}

[data-testid="stSidebar"] {
    background: #fff;
    border-right: 1px solid var(--line);
}

hr { border-color: var(--line) !important; }

@media (max-width: 768px) {
    .block-container { padding: .75rem .85rem 3rem; }
    h1 { font-size: 2.55rem !important; line-height: 1.0 !important; }
    h2 { font-size: 1.55rem !important; }
    h3 { font-size: 1.08rem !important; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 15px !important; }
    [data-testid="stFileUploaderDropzone"] { padding: 1rem; border-radius: 14px; }
    [data-testid="stRadio"] > div { flex-direction: column !important; }
    [data-testid="stRadio"] label { width: 100%; min-height: 46px; }
    .stButton > button, .stDownloadButton > button { width: 100%; min-height: 50px; }
    [data-baseweb="tab-list"] { overflow-x: auto; white-space: nowrap; gap: 1rem; }
    [data-testid="column"] { min-width: 0 !important; }
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
# HERO — simple, Chassly-inspired hierarchy
# ============================================================
st.caption("KNOW YOUR DAMAGE")
st.title("See the damage.\nUnderstand the vehicle.")
st.write(
    "Upload a vehicle photo and get an instant AI assessment with annotated damage, "
    "confidence scores and evidence for every detected area."
)
st.caption("Fast visual assessment · Two detection models · Detailed evidence")

st.write("")
st.write("")

# ============================================================
# ASSESSMENT
# ============================================================
st.subheader("Start an assessment")
st.caption("One image is enough to test the current model. Use a clear, well-lit exterior photo.")
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
    st.info("Upload a vehicle image to begin.")
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

    st.caption(f"ASSESSMENT · {inspection_id}")
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
