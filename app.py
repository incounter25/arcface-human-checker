import streamlit as st
import cv2
from PIL import Image
import numpy as np
import insightface
import joblib

# ArcFace 모델 로딩
model = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0)

# SVM 분류기 로딩 (예외 처리 포함)
try:
    classifier = joblib.load("is_human_classifier.pkl")
except Exception as e:
    st.error(f"❌ 분류기 모델을 불러오지 못했습니다: {e}")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="다중 얼굴 판별기", layout="centered")
st.title("🧠 ArcFace + SVM 다중 얼굴 판별기")

uploaded_file = st.file_uploader("얼굴 이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    faces = model.get(img_bgr)
    if not faces:
        st.error("❌ 얼굴을 감지하지 못했습니다.")
    else:
        for face in faces:
            bbox = face.bbox.astype(int)
            emb = face.embedding.reshape(1, -1)

            pred = classifier.predict(emb)[0]
            proba = classifier.predict_proba(emb)[0][pred]

            if pred == 1:
                label = f"✅ 사람 ({proba:.2f})"
                color = (0, 255, 0)
            else:
                label = f"❌ 비사람 ({proba:.2f})"
                color = (0, 0, 255)

            cv2.rectangle(img_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(img_bgr, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="📷 분석 결과", use_column_width=True)
