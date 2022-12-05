"""
This is amazing tutorial for installing yolov5:
https://wandb.ai/onlineinference/YOLO/reports/YOLOv5-Object-Detection-on-Windows-Step-By-Step-Tutorial---VmlldzoxMDQwNzk4

Documentation for using pythorch hub:
https://docs.ultralytics.com/tutorials/pytorch-hub/
"""

import torch
import cv2

video_path = '../videos/clip_1.mp4'
model_path = '../yolov5s.pt'
video = cv2.VideoCapture(video_path)

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model = torch.hub.load('../../../../yolov5/', 'custom', path=model_path, source='local')

while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break

    # Press q to exit video
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    l = 0
    r = 150
    frame=frame[l:l+640, r:r+640]
    results = model(frame)

    results.render()  # Draws bounding boxes for detections on given image
    cv2.imshow('output', results.ims[0])

    # print("Predictions: ", results.xyxy[0])
    # print(results)
