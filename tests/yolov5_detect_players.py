"""
This is amazing tutorial for installing yolov5:
https://wandb.ai/onlineinference/YOLO/reports/YOLOv5-Object-Detection-on-Windows-Step-By-Step-Tutorial---VmlldzoxMDQwNzk4

Documentation for using pythorch hub:
https://docs.ultralytics.com/tutorials/pytorch-hub/
"""

import torch
import cv2
# from pafy import pafy


# def get_youtube_cap(url):
#     play = pafy.new(url).streams[-1] # we will take the lowest quality stream
#     assert play is not None # makes sure we get an error if the video failed to load
#     return cv2.VideoCapture(play.url)


def get_player_positions(boxes):
    positions = []
    for ((x1, y1), (x2, y2)) in boxes:
        x3 = (x1+x2)/2
        p = 0.9
        y3 = y1 + (y2 - y1) * p
        positions.append((int(x3), int(y3)))
    return positions


def get_player_boxes(detections):
    boxes = []
    results = detections.pandas().xyxy[0].to_dict(orient="records")
    for result in results:
        # con = result['confidence']
        cs = result['class']
        x1 = int(result['xmin'])
        y1 = int(result['ymin'])
        x2 = int(result['xmax'])
        y2 = int(result['ymax'])
        if cs == 0:
            boxes.append(((x1, y1), (x2, y2)))
    return boxes


def main():
    video_path = '../videos/clip_1.mp4'
    model_path = '../models/yolov5s.pt'

    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model = torch.hub.load('../../../../yolov5/', 'custom', path=model_path, source='local')
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
        detections = model(frame)

        boxes = get_player_boxes(detections)
        for ((x1, y1), (x2, y2)) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        positions = get_player_positions(boxes)
        for (x, y) in positions:
            print(x, y)
            cv2.circle(frame, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

        cv2.imshow('output', frame)
        # detections.render()  # Draws bounding boxes for detections on given image
        # cv2.imshow('output', detections.ims[0])


if __name__ == "__main__":
    main()