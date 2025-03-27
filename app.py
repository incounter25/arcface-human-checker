import streamlit as st
import cv2
from PIL import Image
import numpy as np
import insightface
import joblib
import os

# 1. ArcFace 로딩
model = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0)

# 2. SVM 분류기 로딩
classifier = joblib.load("is_human_classifier.pkl")

# 3. Streamlit UI
st.set_page_config(page_title="얼굴 감지 및 사람 판별", layout="centered")
st.title("🧠 얼굴이 진짜 사람인가요? (ArcFace + SVM)")

uploaded_file = st.file_uploader("사진을 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 4. 이미지 불러오기
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 5. 얼굴 감지 + 임베딩 추출
    faces = model.get(img_bgr)
    if not faces:
        st.error("❌ 얼굴을 감지하지 못했습니다.")
    else:
        face = faces[0]
        bbox = face.bbox.astype(int)
        emb = face.embedding.reshape(1, -1)

        # 6. SVM으로 사람인지 예측
        pred = classifier.predict(emb)[0]
        proba = classifier.predict_proba(emb)[0][pred]

        # 7. 시각화용 라벨
        if pred == 1:
            label = f"✅ 사람입니다 (정확도: {proba:.2f})"
            st.success(label)
        else:
            label = f"❌ 사람 얼굴이 아닙니다 (정확도: {proba:.2f})"
            st.warning(label)

        # 8. 이미지에 표시
        cv2.rectangle(img_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(img_bgr, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 9. 결과 출력
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="📷 분석 결과", use_column_width=True)
