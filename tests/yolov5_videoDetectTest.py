"""
This is amazing tutorial for installing yolov5:
https://wandb.ai/onlineinference/YOLO/reports/YOLOv5-Object-Detection-on-Windows-Step-By-Step-Tutorial---VmlldzoxMDQwNzk4

Documentation for using pythorch hub:
https://docs.ultralytics.com/tutorials/pytorch-hub/
"""

import torch
import cv2
from pafy import pafy

video_path = '../videos/clip_1.mp4'
model_path = '../yolov5s.pt'

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model = torch.hub.load('../../../../yolov5/', 'custom', path=model_path, source='local')


def get_youtube_cap(url):
    play = pafy.new(url).streams[-1] # we will take the lowest quality stream
    assert play is not None # makes sure we get an error if the video failed to load
    return cv2.VideoCapture(play.url)


video = cv2.VideoCapture(video_path)
# video = get_youtube_cap("https://www.youtube.com/watch?v=pO1Wt7t04QQ") # KOŠARKA
# video = get_youtube_cap("https://www.youtube.com/watch?v=fQoJZuBwrkU") # FLAPPY BIRD

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

    # print("Predictions: ", results.xyxy[0])
    # print(results)
