import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

# ------------------------------------------------------------
# Page
# ------------------------------------------------------------
st.set_page_config(
    page_title="CarDD Vision | Vehicle Inspection",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS is deliberately de-indented and kept separate from page content.
# All visible content below uses native Streamlit elements, so HTML tags
# can never appear as text in the app.
st.markdown("""
<style>
:root { --brand:#111827; --accent:#2563eb; }
.stApp { background:#f6f7f9; }
[data-testid="stHeader"] { background:rgba(246,247,249,.85); }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility:hidden; }
.block-container { max-width:1240px; padding-top:1.4rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #e5e7eb; }
[data-testid="stSidebar"] .block-container { padding-top:1.2rem; }
[data-testid="stFileUploaderDropzone"] { background:#fff; border:1.5px dashed #cbd5e1; border-radius:16px; padding:1.2rem; }
[data-testid="stMetric"] { background:#fff; border:1px solid #e5e7eb; border-radius:16px; padding:1rem 1.1rem; }
div[data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-radius:18px; }
.stButton > button { border-radius:12px; min-height:46px; font-weight:700; }
.stButton > button[kind="primary"] { background:#111827; border-color:#111827; }
.stButton > button[kind="primary"]:hover { background:#2563eb; border-color:#2563eb; }
[data-testid="stImage"] img { border-radius:14px; }
.small-note { color:#667085; font-size:.84rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------
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
            detections.append({
                "name": damage_name,
                "confidence": score,
                "box": xyxy,
                "crop": get_damage_crop(image, xyxy),
            })

    detections.sort(key=lambda x: x["confidence"], reverse=True)
    return output_image, detections, scan_time


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
for key, default in {
    "inspection_result": None,
    "inspection_source_name": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.title("🚘 CarDD Vision")
    st.caption("AI-assisted vehicle inspection")
    st.divider()

    st.subheader("Inspection settings")
    mode = st.radio(
        "Inspection mode",
        ["Precision inspection", "Quick inspection"],
        help="Precision uses the larger model. Quick inspection uses the lighter model.",
    )

    confidence = st.slider(
        "Detection sensitivity",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Lower values detect more possible damage but may increase false positives.",
    )
    st.caption(f"Minimum confidence: {confidence:.0%}")

    with st.expander("Advanced model details"):
        if mode == "Precision inspection":
            st.write("Model: YOLO11m")
            st.write("Use: higher-quality visual inspection")
        else:
            st.write("Model: YOLOv8s")
            st.write("Use: faster lightweight inspection")
        st.caption("These are technical settings and do not represent damage severity.")

# ------------------------------------------------------------
# Header / product positioning
# ------------------------------------------------------------
st.title("Vehicle damage inspection")
st.caption(
    "Upload a vehicle image to identify visible exterior damage, review AI evidence, "
    "and inspect confidence for every finding."
)

# compact trust row
c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.markdown("**⚡ Fast visual screening**")
        st.caption("Inspect a vehicle photo in seconds.")
with c2:
    with st.container(border=True):
        st.markdown("**🎯 Evidence-based findings**")
        st.caption("See the exact region behind each detection.")
with c3:
    with st.container(border=True):
        st.markdown("**🔎 Confidence included**")
        st.caption("Review model certainty for every finding.")

st.write("")

# ------------------------------------------------------------
# Upload
# ------------------------------------------------------------
with st.container(border=True):
    st.subheader("Start a new inspection")
    uploaded_file = st.file_uploader(
        "Vehicle photo",
        type=["jpg", "jpeg", "png", "webp"],
        help="For best results, use a clear, well-lit exterior vehicle photo.",
    )

    if uploaded_file is None:
        st.info("Upload a vehicle image to begin.", icon="📷")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        preview, action = st.columns([1.5, 1])
        with preview:
            st.image(image, caption="Uploaded vehicle", use_container_width=True)
        with action:
            st.markdown("### Ready to inspect")
            st.write(
                "The system will analyse the visible vehicle area and return the "
                "annotated result, damage categories, confidence scores and evidence crops."
            )
            st.caption(f"Image size: {image.width} × {image.height}px")
            run = st.button("Run vehicle inspection", type="primary", use_container_width=True)

        if run:
            try:
                if mode == "Precision inspection":
                    with st.spinner("Running precision vehicle inspection..."):
                        model = load_yolo11m()
                        output_image, detections, scan_time = run_scan(model, image, confidence)
                    model_name = "YOLO11m"
                else:
                    with st.spinner("Running quick vehicle inspection..."):
                        model = load_yolov8()
                        output_image, detections, scan_time = run_scan(model, image, confidence)
                    model_name = "YOLOv8s"

                st.session_state.inspection_result = {
                    "original": image.copy(),
                    "annotated": output_image,
                    "detections": detections,
                    "scan_time": scan_time,
                    "model_name": model_name,
                    "mode": mode,
                    "threshold": confidence,
                }
                st.session_state.inspection_source_name = uploaded_file.name
            except Exception as exc:
                st.error(f"Inspection failed: {exc}")

# ------------------------------------------------------------
# Results — persist after reruns
# ------------------------------------------------------------
result = st.session_state.inspection_result

if result is not None:
    detections = result["detections"]
    highest = max((d["confidence"] for d in detections), default=0.0)
    unique_types = len({d["name"] for d in detections})

    st.write("")
    st.divider()
    st.subheader("Inspection result")
    st.caption(
        f"{st.session_state.inspection_source_name or 'Vehicle image'} · "
        f"{result['mode']} · threshold {result['threshold']:.0%}"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Detected regions", len(detections))
    m2.metric("Damage types", unique_types)
    m3.metric("Highest confidence", f"{highest:.0%}")
    m4.metric("Scan time", f"{result['scan_time']:.2f}s")

    st.write("")
    st.markdown("### Original vs AI inspection")
    before, after = st.columns(2)
    with before:
        st.image(result["original"], caption="Original vehicle", use_container_width=True)
    with after:
        st.image(result["annotated"], caption="AI inspection output", use_container_width=True)

    st.write("")
    st.markdown("### Detailed findings")

    if not detections:
        st.success(
            "No visible damage was detected above the selected confidence threshold.",
            icon="✅",
        )
    else:
        # Full summary table
        summary_rows = []
        for i, d in enumerate(detections, start=1):
            x1, y1, x2, y2 = d["box"]
            summary_rows.append({
                "#": i,
                "Damage": d["name"],
                "Confidence": f"{d['confidence']:.1%}",
                "Box (x1, y1, x2, y2)": f"{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        st.write("")
        for i, d in enumerate(detections, start=1):
            with st.container(border=True):
                left, right = st.columns([1.2, 1])
                with left:
                    st.markdown(f"#### {i:02d}. {d['name']}")
                    st.metric("Detection confidence", f"{d['confidence']:.1%}")
                    st.progress(min(max(d["confidence"], 0.0), 1.0))
                    x1, y1, x2, y2 = d["box"]
                    st.write(f"**Bounding box:** ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})")
                    st.caption(
                        "Confidence is the model's certainty about the damage category. "
                        "It is not a measure of repair cost, physical severity or vehicle safety."
                    )
                with right:
                    st.image(d["crop"], caption=f"Evidence region {i}", use_container_width=True)

    with st.expander("Technical inspection details"):
        st.write(f"**Model:** {result['model_name']}")
        st.write(f"**Inspection mode:** {result['mode']}")
        st.write(f"**Confidence threshold:** {result['threshold']:.0%}")
        st.write(f"**Inference time:** {result['scan_time']:.3f} seconds")
        st.write(f"**Image dimensions:** {result['original'].width} × {result['original'].height}px")

    st.warning(
        "AI-assisted visual screening only. Results should be reviewed by a qualified human inspector, "
        "especially for safety, valuation or repair decisions.",
        icon="⚠️",
    )
