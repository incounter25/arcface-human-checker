import gradio as gr
import cv2
import numpy as np
from PIL import Image
import insightface
import joblib

# ArcFace 얼굴 분석기 로딩
face_model = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_model.prepare(ctx_id=0)

# SVM 모델 로드
clf = joblib.load("is_human_classifier.pkl")

# 얼굴 판별 함수
def predict(image):
    img = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    faces = face_model.get(img_bgr)
    if not faces:
        return "❌ 얼굴이 감지되지 않았습니다."

    face = faces[0]
    emb = face.embedding.reshape(1, -1)
    pred = clf.predict(emb)[0]
    proba = clf.predict_proba(emb)[0][pred]
    result = "✅ 사람입니다" if pred == 1 else "❌ 사람이 아닙니다"
    return f"{result} (정확도: {proba:.2f})"

# Gradio 인터페이스 정의
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(source="webcam", label="웹캠으로 얼굴을 촬영하세요"),
    outputs=gr.Textbox(label="판별 결과"),
    live=True,
    title="👤 실시간 얼굴 판별기",
    description="웹캠을 통해 얼굴을 인식하고 사람이 맞는지 판별합니다."
)

if __name__ == "__main__":
    demo.launch()
