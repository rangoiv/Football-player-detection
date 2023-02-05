
import torch
import cv2

video_path = '../videos/clip_1.mp4'
yolov5_path = 'C:/Users/Rango/yolov5'
model_path = '../models/goals.pt'

model = torch.hub.load(yolov5_path, 'custom', path=model_path, source='local')

video = cv2.VideoCapture(video_path)

while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break

    # Press q to exit video
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    results = model(frame)

    results.render()  # Draws bounding boxes for detections on given image
    cv2.imshow('output', results.ims[0])
