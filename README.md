# 🚗 Vehicle Instance Segmentation with YOLOv9

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv9-orange.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)

This repository contains a complete pipeline for fine-tuning and evaluating the **YOLOv9** model (`yolov9c-seg`) for **Vehicle Instance Segmentation** using precise polygon annotation coordinates. The goal is to detect vehicles and delineate their exact boundaries for intelligent transportation systems and autonomous driving.

---

## 📊 Dataset Summary
The dataset was loaded and preprocessed via [Roboflow](https://roboflow.com/) in YOLO Segmentation format.
* **Train Set:** 7,545 images
* **Validation Set:** 587 images
* **Test Set:** 193 images
* **Classes (5 Categories):** `car`, `truck`, `bus`, `bicycle`, `motorcycle`

---

## 🚀 Key Features & Pipeline
1. **Dataset Loading & Preprocessing:** Automated dataset download from Roboflow, verification of label files, and handling of image-label pairs.
2. **Exploratory Data Analysis (EDA):** Extraction of polygon coordinates to visualize ground truth bounding polygons, binary masks, and class distributions.
3. **Model Training:** Fine-tuning `yolov9c-seg` with Multi-GPU support, Automatic Mixed Precision (AMP), and custom hyperparameters.
4. **Advanced Evaluation:** Evaluation using standard detection/segmentation metrics (`mAP50`, `mAP50-95`) alongside custom segmentation metrics like **Dice Coefficient** and **Dice Loss**.

---

## 📈 Results & Performance
* **Mask mAP50:** ~0.749
* **Mask mAP50-95:** ~0.698
* **Mean Dice Coefficient (Test Set):** ~0.7997

---

## 🔗 Links & Resources
* 📓 **Kaggle Notebook:** [Vehicle Instance Segmentation with YOLOv9 Notebook](https://www.kaggle.com/code/samoura/vehicle-instance-segmentation-with-yolov9)
* 🌐 **Streamlit App:** [Streamlit Application](https://vehicle-instance-segmentation.streamlit.app/)

---

## 📁 Repository Structure

```text
├── app.py              # Web deployment interface with automated model download
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

```
