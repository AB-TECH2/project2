import streamlit as st
import numpy as np
import tensorflow as tf

tflite = tf.lite.Interpreter
from PIL import Image
import os
import time
import random
from fpdf import FPDF
from datetime import datetime

st.set_page_config(page_title="SkinScan Pro 2025", page_icon="🩺", layout="wide")

# اللغة
language = st.sidebar.selectbox("🌐 Language / اللغة", ["العربية", "English"])
lang = "ar" if language == "العربية" else "en"

# الوضع الليلي / الفاتح
theme = st.sidebar.radio("🎨 اختر النمط", ["Light - فاتح", "Dark - غامق"])
is_dark = theme == "Dark - غامق"

# النصوص حسب اللغة
texts = {
    "ar": {
        "title": "🩺 أهلاً بك في تطبيق", "app_name": "SkinScan Pro 2025",
        "desc": "تشخيص ذكي واحترافي للأمراض الجلدية باستخدام الذكاء الاصطناعي",
        "upload_option": "اختار طريقة رفع الصورة:", "upload_file": "📤 رفع صورة من الجهاز", "upload_camera": "📷 التقاط من الكاميرا",
        "upload_prompt": "اختر صورة لمرض جلدي", "image_caption": "📸 الصورة التي تم تحليلها", "loading": "جاري تحليل الصورة ...",
        "diagnosis": "✅ التشخيص", "confidence": "نسبة الثقة", "treatment": "💊 العلاج المقترح:",
        "no_result": "⚠️ لم يتم التعرف على المرض. الرجاء تجربة صورة أوضح", "pdf_button": "📥 حفظ النتيجة كـ PDF",
        "pdf_download": "📄 تحميل النتيجة PDF", "retry": "🔄 إعادة التشخيص", "real_image": "🔍 صورة حقيقية لـ",
        "about_title": "📘 عن التطبيق / About"
    },
    "en": {
        "title": "🩺 Welcome to", "app_name": "SkinScan Pro 2025",
        "desc": "Smart and professional skin disease diagnosis using AI",
        "upload_option": "Choose image upload method:", "upload_file": "📤 Upload from device", "upload_camera": "📷 Take a photo",
        "upload_prompt": "Upload a skin disease image", "image_caption": "📸 Image to be analyzed", "loading": "Analyzing image...",
        "diagnosis": "✅ Diagnosis", "confidence": "Confidence", "treatment": "💊 Suggested Treatment:",
        "no_result": "⚠️ Disease not recognized. Please try a clearer image.", "pdf_button": "📥 Save result as PDF",
        "pdf_download": "📄 Download PDF result", "retry": "🔄 Retake diagnosis", "real_image": "🔍 Real image of",
        "about_title": "📘 About the app"
    }
}
T = texts[lang]

# CSS احترافي
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&family=Poppins:wght@400;700&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Cairo', sans-serif;
        background: linear-gradient(120deg, #fceff9, #e0f7fa, #f3f9ff);
        background-size: 400% 400%;
        animation: backgroundFlow 10s ease infinite;
        color: {'#fff' if is_dark else '#000'};
    }}
    @keyframes backgroundFlow {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    .stButton>button {{
        background-color: #00b4d8;
        color: white;
        font-weight: bold;
        font-family: 'Poppins', sans-serif;
        border-radius: 12px;
        padding: 12px 24px;
        transition: all 0.3s ease-in-out;
    }}
    .stButton>button:hover {{
        background-color: #0077b6;
        transform: scale(1.07);
    }}
    .result-box {{
        background: linear-gradient(135deg, #ffffff, #e3f6ff);
        border-radius: 20px;
        padding: 25px;
        margin-top: 25px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.12);
        animation: zoomIn 0.5s ease;
    }}
    img {{
        border-radius: 14px !important;
        transition: all 0.3s ease-in-out;
    }}
    img:hover {{
        transform: scale(1.03);
    }}
    h3::before {{ content: "🩺 "; }}
    h4::before {{ content: "💊 "; }}
    @keyframes zoomIn {{
        from {{ transform: scale(0.95); opacity: 0; }}
        to {{ transform: scale(1); opacity: 1; }}
    }}
    </style>
""", unsafe_allow_html=True)

# عرض ترحيبي
st.markdown(f"<h2 style='text-align:center'>{T['title']}</h2><h1 style='text-align:center'>{T['app_name']}</h1><h3 style='text-align:center'>{T['desc']}</h3><hr>", unsafe_allow_html=True)

# تحميل النموذج
interpreter = tflite.Interpreter(model_path="skin_disease_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# قاعدة بيانات الأمراض (أمثلة فقط - كملها برحتك)
disease_info = {
    "Eczema": {"ar": "الأكزيما", "treatment": "ترطيب الجلد، كريمات الستيرويد"},
    "Psoriasis": {"ar": "الصدفية", "treatment": "علاج ضوئي، أدوية موضعية"},
    "Acne": {"ar": "حب الشباب", "treatment": "كريمات موضعية، مضاد حيوي"},
    "Impetigo": {"ar": "القوباء", "treatment": "مضاد حيوي موضعي"},
    "Scabies": {"ar": "الجرب", "treatment": "بيرمثرين وغسل المفروشات"},
    "Melasma": {"ar": "الكلف", "treatment": "تفتيح البشرة وواقي شمس"},
    "Warts": {"ar": "الثآليل", "treatment": "تجميد أو ليزر"},
    "Vitiligo": {"ar": "البهاق", "treatment": "كريمات وتعرض للشمس"},
    # كمل باقي الأمراض هنا...
}

# تابعني للجزء 2 فيه: الرفع - التشخيص - العرض - PDF - البحث - صور
# دالة التصنيف
def classify_image(img: Image.Image):
    img = img.resize((224, 224))
    input_data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_index = int(np.argmax(output_data))
    confidence = float(np.max(output_data))
    classes = list(disease_info.keys())
    predicted_disease = classes[predicted_index % len(classes)]
    return predicted_disease, confidence

# رفع صورة
option = st.radio(T["upload_option"], [T["upload_file"], T["upload_camera"]])
uploaded_img = None
if option == T["upload_file"]:
    uploaded_img = st.file_uploader(T["upload_prompt"], type=["jpg", "jpeg", "png"])
else:
    uploaded_img = st.camera_input(T["upload_prompt"])

# معالجة الصورة
if uploaded_img:
    image = Image.open(uploaded_img).convert("RGB")
    st.image(image, caption=T["image_caption"], use_column_width=True)

    with st.spinner(T["loading"]):
        time.sleep(1.5)
        predicted_disease, confidence = classify_image(image)

    if predicted_disease in disease_info:
        data = disease_info[predicted_disease]
        st.markdown(f"""
        <div class='result-box'>
            <h3>{T['diagnosis']}: {data['ar']} ({predicted_disease})</h3>
            <p><strong>{T['confidence']}:</strong> {confidence*100:.2f}%</p>
            <h4>{T['treatment']}</h4>
            <p>{data['treatment']}</p>
        </div>
        """, unsafe_allow_html=True)

        # صور من المجلد
        folder = os.path.join("skin_dataset", predicted_disease)
        if os.path.isdir(folder):
            files = [f for f in os.listdir(folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
            if files:
                st.markdown(f"### {T['real_image']} {data['ar']}")
                selected_imgs = random.sample(files, min(5, len(files)))
                img_rows = [selected_imgs[:3], selected_imgs[3:]]
                for row in img_rows:
                    cols = st.columns(len(row))
                    for i, file in enumerate(row):
                        img_path = os.path.join(folder, file)
                        cols[i].image(img_path, use_column_width=True)

        # ترشيحات لو الثقة ضعيفة
        if confidence < 0.7:
            st.warning("🔍 احتمالات أخرى مشابهة:")
            suggestions = random.sample([d for d in disease_info if d != predicted_disease], 2)
            for s in suggestions:
                st.markdown(f"- {disease_info[s]['ar']} ({s})")

        # حفظ PDF
        if st.button(T["pdf_button"]):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(200, 10, txt="SkinScan Pro 2025", ln=True, align="C")
            pdf.ln(10)
            pdf.cell(200, 10, txt=f"{T['diagnosis']}: {data['ar']} ({predicted_disease})", ln=True)
            pdf.cell(200, 10, txt=f"{T['confidence']}: {confidence*100:.2f}%", ln=True)
            pdf.multi_cell(0, 10, txt=f"{T['treatment']}:\n{data['treatment']}")
            pdf.cell(200, 10, txt=f"تاريخ التحليل: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
            pdf.output("result.pdf")
            with open("result.pdf", "rb") as f:
                st.download_button(T["pdf_download"], data=f, file_name="SkinScan_Result.pdf", mime="application/pdf")

        if st.button(T["retry"]):
            st.experimental_rerun()
    else:
        st.error(T["no_result"])

# 🔍 البحث عن مرض
search_term = st.sidebar.text_input("🔍 ابحث عن مرض بالإنجليزي أو العربي")
if search_term:
    match = None
    for key, val in disease_info.items():
        if search_term.lower() in key.lower() or search_term in val.get("ar", ""):
            match = (key, val)
            break

    if match:
        key, val = match
        st.markdown(f"""
        <div class='result-box'>
            <h3>{T['diagnosis']}: {val['ar']} ({key})</h3>
            <h4>{T['treatment']}</h4>
            <p>{val['treatment']}</p>
        </div>
        """, unsafe_allow_html=True)

        folder = os.path.join("skin_dataset", key)
        if os.path.isdir(folder):
            files = [f for f in os.listdir(folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
            if files:
                st.markdown(f"### {T['real_image']} {val['ar']}")
                selected_imgs = random.sample(files, min(5, len(files)))
                img_rows = [selected_imgs[:3], selected_imgs[3:]]
                for row in img_rows:
                    cols = st.columns(len(row))
                    for i, file in enumerate(row):
                        img_path = os.path.join(folder, file)
                        cols[i].image(img_path, use_column_width=True)
    else:
        st.sidebar.warning("❌ لم يتم العثور على المرض.")

# 📘 عن التطبيق
with st.expander(T["about_title"]):
    st.markdown("""
    <div style='background-color:#f0f8ff; padding: 20px; border-radius: 12px; box-shadow: 0px 2px 8px rgba(0,0,0,0.1);'>
        <h2 style='color:#0077b6;'>💡 SkinScan Pro 2025</h2>
        <ul>
            <li>🧠 يعتمد على نموذج TFLite للتشخيص الذكي.</li>
            <li>📷 يدعم رفع الصور أو استخدام الكاميرا.</li>
            <li>📄 يحفظ النتيجة كـ PDF احترافي.</li>
            <li>🌗 يدعم الوضع الليلي واللغتين.</li>
        </ul>
        <h4>👨‍💻 تم التطوير بواسطة:</h4>
        <p><strong>عبدالرحمن العتر</strong><br>كلية التربية النوعية – جامعة دمياط<br>بإشراف: د/ رانيا أبوجلالة</p>
    </div>
    """, unsafe_allow_html=True)
