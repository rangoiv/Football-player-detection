
import cv2
import torch

from tests.concat_image import concat
from tests.cv2_out_lines_detection import get_top_lines, assign_sides
from tests.yolov5_detect_goal import get_goal_box
from tests.yolov5_detect_players import get_player_boxes, get_player_positions


def main():
    video_path = '../videos/clip_1.mp4'
    pitch_image_path = '../images/pitch.png'
    yolov5_path = 'C:/Users/Rango/yolov5'
    goal_model_path = '../models/goals.pt'
    player_model_path = '../models/yolov5s.pt'

    goal_model = torch.hub.load(yolov5_path, 'custom', path=goal_model_path, source='local')
    player_model = torch.hub.load(yolov5_path, 'custom', path=player_model_path, source='local')

    pitch_image = cv2.imread(pitch_image_path)
    pitch_image = cv2.resize(pitch_image,
                             (int(pitch_image.shape[1] * 1.3), int(pitch_image.shape[0] * 1.3)),
                             interpolation=cv2.INTER_AREA)
    video = cv2.VideoCapture(video_path)

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        # Press q to exit video
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # Get positions
        goal_detections = goal_model(frame)
        box = get_goal_box(goal_detections)

        player_detections = player_model(frame)
        boxes = get_player_boxes(player_detections)

        lines = get_top_lines(frame)
        lines = assign_sides(lines)


        # Draw Goal
        if box != -1:
            (x1, y1), (x2, y2) = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw Players
        for ((x1, y1), (x2, y2)) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        positions = get_player_positions(boxes)
        for (x, y) in positions:
            cv2.circle(frame, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

        # Draw Lines
        for ((x1, y1), (x2, y2), s) in lines:
            if s == 't':
                col = (0, 0, 255)
            elif s == 'l':
                col = (0, 255, 0)
            else:
                col = (255, 0, 0)
            cv2.line(frame, (x1, y1), (x2, y2), col, 2)

        concated = concat(frame, pitch_image, 'h')
        cv2.imshow('Match Detection', concated)


if __name__ == "__main__":
    main()
