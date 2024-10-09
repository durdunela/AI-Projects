import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('/home/mari/Development/AI/yolo-people-counting/runs/detect/train/weights/best.pt')

video_path = 'Detect Players/test2.webm'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True, show=False, tracker="botsort.yaml")
        annotated_frame = results[0].plot()
        
        cv2.imshow('YOLO Tracking', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
