import streamlit as st
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Car Damage Detector",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Damage Detector")
st.caption("Compare AI models for car damage detection and segmentation.")

# -----------------------------
# MODEL LOADERS
# -----------------------------

@st.cache_resource
def load_yolov8():
    model_path = hf_hub_download(
        repo_id="abdullahg7/cardd-yolov8s",
        filename="v2.0/best.pt"
    )
    return YOLO(model_path)


@st.cache_resource
def load_yolo11():
    model_path = hf_hub_download(
        repo_id="harpreetsahota/car-dd-segmentation-yolov11",
        filename="best.pt"
    )
    return YOLO(model_path)


# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Choose AI Model",
    [
        "YOLOv8s — Fast",
        "YOLO11 — High Accuracy"
    ]
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

# -----------------------------
# IMAGE UPLOAD
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload a car image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("🔍 Detect Damage", use_container_width=True):

        with st.spinner("Loading AI model and analyzing image..."):

            if model_choice == "YOLOv8s — Fast":
                model = load_yolov8()
            else:
                model = load_yolo11()

            results = model.predict(
                source=np.array(image),
                conf=confidence,
                verbose=False
            )

            result_image = results[0].plot()

        with col2:
            st.subheader("Detection Result")
            st.image(result_image, use_container_width=True)

        st.divider()

        st.subheader("🔎 Detected Damage")

        if results[0].boxes is not None and len(results[0].boxes) > 0:

            for box in results[0].boxes:

                class_id = int(box.cls[0])
                score = float(box.conf[0])

                damage_type = model.names[class_id]

                st.write(
                    f"🔴 **{damage_type.replace('_', ' ').title()}** "
                    f"— Confidence: **{score:.1%}**"
                )

        else:
            st.success("No damage detected.")