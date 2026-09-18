import os
import gdown
import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO

# Google Drive File Configuration
DRIVE_FILE_ID = "1beRlmhl1fj-eS7W6jv2awfXNHad_OoHz"
MODEL_PATH = "best.pt"

def download_model():
    """Downloads model weights from Google Drive if not locally present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model weights from Google Drive...")
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Model weights downloaded successfully.")

# Initialize Model
download_model()
model = YOLO(MODEL_PATH)

def segment_vehicles(image, conf_threshold):
    """Performs YOLOv9 instance segmentation on input images."""
    if image is None:
        return None
    
    # Run Inference
    results = model.predict(
        source=image,
        conf=conf_threshold,
        retina_masks=True,
        verbose=False
    )
    
    # Render Output Mask Overlay
    res_plotted = results[0].plot()
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    return res_rgb

# Gradio Interface Build
demo = gr.Interface(
    fn=segment_vehicles,
    inputs=[
        gr.Image(type="numpy", label="Upload Vehicle Image"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.25, step=0.05, label="Confidence Threshold")
    ],
    outputs=gr.Image(type="numpy", label="Segmentation Output"),
    title="🚗 Vehicle Instance Segmentation (YOLOv9)",
    description="Upload an image to segment vehicles across 5 categories: Car, Bus, Truck, Bicycle, and Motorcycle."
)

if __name__ == "__main__":
    demo.launch()
