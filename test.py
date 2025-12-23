from ultralytics import YOLO

model = YOLO('yolo11s.pt')
model.predict(source=0, show=True, conf=0.35)  # source=0 is default webcam