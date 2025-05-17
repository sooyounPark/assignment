from ultralytics import YOLO
from pathlib import Path
import os

# 1. 경로 설정
DATA_YAML_PATH = '/Users/suyeon/Library/CloudStorage/GoogleDrive-ekvlsp8819@gmail.com/내 드라이브/학교/대학원/수업/지능화 캡스톤/12주차과제/My First Project.v2i.yolov8/data.yaml'
MODEL_NAME = 'yolov8n.pt'  # 또는 yolov8s.pt, yolov9n.pt 등

# 2. 모델 로드 및 학습
model = YOLO(MODEL_NAME)
model.train(
    data=DATA_YAML_PATH,
    epochs=50,
    imgsz=640,
    batch=8,
    name='website_screenshot_yolo'
)

# 3. 검증
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")

# 4. 테스트 이미지 예측 및 결과 저장
TEST_IMAGE_DIR = '/Users/suyeon/Library/CloudStorage/GoogleDrive-ekvlsp8819@gmail.com/내 드라이브/학교/대학원/수업/지능화 캡스톤/12주차과제/My First Project.v2i.yolov8/test/images'
OUTPUT_DIR = 'runs/detect/website_screenshot_test'
os.makedirs(OUTPUT_DIR, exist_ok=True)

trained_model = YOLO('runs/detect/website_screenshot_yolo/weights/best.pt')
for img_path in Path(TEST_IMAGE_DIR).glob('*.jpg'):
    trained_model(img_path, save=True, save_txt=True, save_conf=True, project='runs/detect', name='website_screenshot_test')
    print(f'Processed: {img_path.name}')