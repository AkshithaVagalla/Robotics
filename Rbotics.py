import cv2
import numpy as np
from ultralytics import YOLO
import time


# load a pretrained YOLOv8n model
model = YOLO("weights/yolov5s.pt", "v5")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

prev_frame_time = 0
new_frame_time = 0
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    new_frame_time = time.time()
    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.65, save=False)

    DP = detect_params[0].numpy()
    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0), 3)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    fps = int(fps)
    fps = str(fps)
    cv2.putText(
        frame, fps,
        (20,20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2
    )
    # Display the resulting frame
    cv2.imshow("ObjectDetection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
