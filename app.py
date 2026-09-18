import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import gdown

# إعدادات الصفحة
st.set_page_config(
    page_title="Vehicle Instance Segmentation",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Vehicle Instance Segmentation")
st.write("قم برفع صورة للكشف عن المركبات وتحديد حدودها بدقة باستخدام نموذج YOLOv9.")

# تحميل النموذج تلقائياً من Google Drive
@st.cache_resource
def load_model():
    model_path = "best.pt"
    
    # تحميل الملف من Google Drive إذا لم يكن موجوداً محلياً
    if not os.path.exists(model_path):
        file_id = "1beRlmhl1fj-eS7W6jv2awfXNHad_OoHz"
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner("جاري تحميل ملف الوزن الخاص بالموديل من Google Drive..."):
            gdown.download(url, model_path, quiet=False)
            
    return YOLO(model_path)

model = load_model()

# واجهة رفع الصور
uploaded_file = st.file_uploader("اختر صورة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="الصورة المرفوعة", use_column_width=True)
    
    if st.button("تشغيل التقسيم (Segmentation)"):
        with st.spinner("جاري المعالجة..."):
            img_array = np.array(image)
            results = model.predict(source=img_array, conf=0.25)
            
            # عرض النتائج
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="نتيجة التقسيم", use_column_width=True)
            st.success("تم التقسيم بنجاح!")
