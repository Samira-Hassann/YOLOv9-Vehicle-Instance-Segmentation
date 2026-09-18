import os
import gdown
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Google Drive File Configuration
DRIVE_FILE_ID = "1beRlmhl1fj-eS7W6jv2awfXNHad_OoHz"
MODEL_PATH = "best.pt"

# إعدادات الصفحة
st.set_page_config(
    page_title="Vehicle Instance Segmentation (YOLOv9)",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Vehicle Instance Segmentation (YOLOv9)")
st.write("Upload an image to segment vehicles across 5 categories: Car, Bus, Truck, Bicycle, and Motorcycle.")

# تحميل وزن الموديل تلقائياً من Google Drive
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights from Google Drive..."):
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    return YOLO(MODEL_PATH)

model = load_model()

# شريط ضبط نسبة الثقة (Confidence Threshold)
conf_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05
)

# رفع الصورة
uploaded_file = st.file_uploader("Upload Vehicle Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Run Segmentation"):
        with st.spinner("Processing..."):
            img_array = np.array(image)
            
            # تشغيل التقسيم باستخدام YOLOv9
            results = model.predict(
                source=img_array,
                conf=conf_threshold,
                retina_masks=True,
                verbose=False
            )
            
            # رسم وتجهيز النتائج
            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            st.image(res_rgb, caption="Segmentation Output", use_column_width=True)
            st.success("Segmentation Completed!")
