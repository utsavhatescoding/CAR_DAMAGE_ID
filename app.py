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
st.write("Upload a car image to detect possible damage.")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="abdullahg7/cardd-yolov8s",
        filename="v2.0/best.pt"
    )

    return YOLO(model_path)


uploaded_file = st.file_uploader(
    "Upload a car image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    if st.button("🔍 Detect Damage"):

        with st.spinner("Loading model and analyzing image..."):

            model = load_model()

            results = model.predict(
                source=np.array(image),
                conf=0.25,
                verbose=False
            )

            result_image = results[0].plot()

        with col2:
            st.subheader("Detection Result")
            st.image(result_image, use_container_width=True)

        st.subheader("Detected Damage")

        if results[0].boxes is not None and len(results[0].boxes) > 0:

            for box in results[0].boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                damage_type = model.names[class_id]

                st.write(
                    f"🔴 **{damage_type.replace('_', ' ').title()}** "
                    f"— {confidence:.1%}"
                )

        else:
            st.success("No damage detected.")