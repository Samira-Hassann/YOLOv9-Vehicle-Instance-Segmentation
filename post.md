How I built an end-to-end Vehicle Instance Segmentation App with YOLOv9 & Streamlit 🚗⚡

Detecting vehicles is good, but accurately segmenting their exact boundaries in real-time is where Computer Vision gets really interesting. 

I set out to build a full pipeline that handles everything from raw annotations to cloud deployment:

🎯 1. Data & Preprocessing
Working with a Roboflow dataset of 8,000+ images across 5 classes (Car, Bus, Truck, Motorcycle, Bicycle), I converted complex polygon coordinates into binary masks to ensure precise ground-truth labels before model training.

⚡ 2. Fine-Tuning YOLOv9 (yolov9c-seg)
Fine-tuned the architecture to achieve high-precision instance masks and fast inference across all vehicle categories.

🌐 3. Interactive Web Deployment
Instead of keeping the model inside a notebook, I built an interactive Streamlit Web App. It automatically fetches the model weights from Google Drive and processes both images and live video streams seamlessly.

---

📌 Explore the Project Resources:

* 🚀 Live Web App: https://vehicle-instance-segmentation.streamlit.app/
* 📂 GitHub Repo: https://github.com/Samira-Hassann/YOLOv9-Vehicle-Instance-Segmentation
* 📓 Kaggle Notebook: https://www.kaggle.com/code/samoura/vehicle-instance-segmentation-with-yolov9

---

🛠️ Tech Stack: Python | PyTorch | YOLOv9 | Ultralytics | OpenCV | Streamlit | Roboflow

#ComputerVision #DeepLearning #YOLOv9 #InstanceSegmentation #Streamlit #PyTorch #MachineLearning #AutonomousVehicles #DataScience
