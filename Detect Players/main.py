import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.yaml")

model = YOLO("yolo11n.pt")

train_results = model.train(
    data="Detect Players/config.yaml",  
    epochs=5,  
    imgsz=640,  
    device="cpu",  
)

results = model.val()

success = model.export(format="onnx")