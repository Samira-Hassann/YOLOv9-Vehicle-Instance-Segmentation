# 🚗 Vehicle Instance Segmentation with YOLOv9

An end-to-end instance segmentation solution using fine-tuned **YOLOv9** (`yolov9c-seg`) to segment 5 vehicle classes (*Car, Bus, Truck, Bicycle, Motorcycle*).

🚀 **Live Demo:** [Vehicle Segmentation Web App](https://vehicle-instance-segmentation.streamlit.app/)  
📓 **Kaggle Notebook:** [Vehicle Instance Segmentation with YOLOv9](https://www.kaggle.com/code/samoura/vehicle-instance-segmentation-with-yolov9)

---

## 📊 Project Overview

- **Model Architecture:** YOLOv9c-seg (Instance Segmentation)
- **Classes:** `Car`, `Bus`, `Truck`, `Bicycle`, `Motorcycle`
- **Performance:** Achieved **~0.80 Mean Dice Score** on test data.
- **Deployment:** Interactive web interface with automated model loading.

---

## 📁 Repository Structure

```text
├── app.py              # Web deployment interface with automated model download
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
