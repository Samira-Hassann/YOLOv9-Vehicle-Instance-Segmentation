import os
import gdown
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# =========================
# Configuration
# =========================

DRIVE_FILE_ID = "1beRlmhl1fj-eS7W6jv2awfXNHad_OoHz"
MODEL_PATH = "best.pt"


# =========================
# Download Model
# =========================

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model weights...")

        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

        gdown.download(
            url,
            MODEL_PATH,
            quiet=False
        )


# =========================
# Load Model
# =========================

@st.cache_resource
def load_model():
    download_model()
    return YOLO(MODEL_PATH)


model = load_model()


# =========================
# Streamlit UI
# =========================

st.title("🚗 Vehicle Instance Segmentation")

st.write(
    "Upload an image to segment vehicles across 5 categories: "
    "Car, Bus, Truck, Bicycle, and Motorcycle."
)


uploaded_file = st.file_uploader(
    "Upload Vehicle Image",
    type=["jpg", "jpeg", "png"]
)


conf_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.25,
    step=0.05
)


# =========================
# Prediction
# =========================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )


    if st.button("Run Segmentation"):

        with st.spinner("Processing..."):

            results = model.predict(
                source=image,
                conf=conf_threshold,
                retina_masks=True,
                verbose=False
            )

            result_image = results[0].plot()


        st.subheader("Segmentation Result")

        st.image(
            result_image,
            caption="Vehicle Segmentation",
            channels="BGR",
            width="stretch"
        )
```
