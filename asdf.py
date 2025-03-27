import cv2
import matplotlib.pyplot as plt

# 얼굴 인식 모델 로딩
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 이미지 로딩
img = cv2.imread("front.jpg")
if img is None:
    raise FileNotFoundError("❌ 이미지가 로드되지 않았습니다.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

print("✅ 감지된 얼굴 수:", len(faces))

# 얼굴 테두리 그리기
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# 이미지 시각화 (matplotlib)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title("Detected Faces")
plt.axis("off")
plt.show()

#streamlit run app.py

