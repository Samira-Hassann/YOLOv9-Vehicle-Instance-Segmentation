import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import gdown
import os

# Page configuration
st.set_page_config(
    page_title="Vehicle Instance Segmentation",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Vehicle Instance Segmentation with YOLOv9")
st.write("Upload an image to perform precise vehicle instance segmentation using fine-tuned YOLOv9.")

# Google Drive File Config
DRIVE_FILE_ID = "1beRlmhl1fj-eS7W6jv2awfXNHad_OoHz"
MODEL_PATH = "yolov9c-seg-vehicle.pt"

@st.cache_resource
def load_model():
    """Download model weights from Google Drive and load model into memory."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights from Google Drive..."):
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    
    # Load YOLO segmentation model
    model = YOLO(MODEL_PATH)
    return model

try:
    model = load_model()
    st.sidebar.success("Model loaded successfully! ✅")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# Sidebar parameters
st.sidebar.header("Model Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold (NMS)", 0.1, 1.0, 0.45, 0.05)

# File uploader
uploaded_file = st.file_uploader("Choose an image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        # Updated argument here: use_container_width=True
        st.image(image, use_container_width=True)

    # Segmentation trigger
    if st.button("Run Segmentation"):
        with st.spinner("Processing image..."):
            # Predict masks and bounding boxes
            results = model.predict(
                source=img_array,
                conf=conf_threshold,
                iou=iou_threshold,
                task="segment"
            )

            res = results[0]
            res_plotted = res.plot()

        with col2:
            st.subheader("Segmentation Result")
            # Updated argument here: use_container_width=True
            st.image(res_plotted, use_container_width=True)

        # Display detection summary
        st.markdown("---")
        st.subheader("📊 Detected Objects Summary")
        
        if res.boxes is not None and len(res.boxes) > 0:
            classes_detected = [model.names[int(cls)] for cls in res.boxes.cls.cpu().numpy()]
            counts = {name: classes_detected.count(name) for name in set(classes_detected)}
            
            for item, count in counts.items():
                st.write(f"- **{item.capitalize()}**: {count}")
        else:
            st.info("No vehicles or targets detected in this image.")
