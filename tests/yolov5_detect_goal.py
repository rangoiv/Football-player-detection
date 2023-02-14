
import torch
import cv2


def get_goal_box(detections):
    boxes = []
    results = detections.pandas().xyxy[0].to_dict(orient="records")
    for result in results:
        con = result['confidence']
        # cs = result['class']
        x1 = int(result['xmin'])
        y1 = int(result['ymin'])
        x2 = int(result['xmax'])
        y2 = int(result['ymax'])
        boxes.append((con, (x1, y1), (x2, y2)))
    if len(boxes) == 0:
        return -1
    boxes.sort(reverse=True)
    _, (x1, y1), (x2, y2) = boxes[0]
    return (x1, y1), (x2, y2)


def main():
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

        detections = model(frame)
        box = get_goal_box(detections)
        if box != -1:
            (x1, y1), (x2, y2) = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imshow('Match Detection', frame)

        # detections.render()  # Draws bounding boxes for detections on given image
        # cv2.imshow('output', detections.ims[0])


if __name__ == "__main__":
    main()
