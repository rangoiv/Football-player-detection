
import cv2
import torch

from concat_image import concat
from cv2_out_lines_detection import get_top_lines, assign_sides
from yolov5_detect_goal import get_goal_box
from yolov5_detect_players import get_player_boxes, get_player_positions
from cam_info import average_player_width, average_player_height, get_field_of_view_horizontal, get_field_of_view_vertical, get_scaled
from player_position_on_map import corner_position, scaled_point, player_pitch_position

def main():
    video_path = '../videos/clip_1.mp4'
    pitch_image_path = '../images/pitch.png'
    yolov5_path = 'C:/leki_autist/umjetna_drugi_projekt/yolov5-master'
    goal_model_path = '../models/goals.pt'
    player_model_path = '../models/yolov5s.pt'

    goal_model = torch.hub.load(yolov5_path, 'custom', path=goal_model_path, source='local')
    player_model = torch.hub.load(yolov5_path, 'custom', path=player_model_path, source='local')

    video = cv2.VideoCapture(video_path)

    frame_number = 1
    fov_horizontal = 0
    fov_vertical = 0

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

        # map for drawing players
        pitch_image = cv2.imread(pitch_image_path)
        pitch_image = cv2.resize(pitch_image,
                             (int(pitch_image.shape[1]), int(pitch_image.shape[0])),
                             interpolation=cv2.INTER_AREA)
        map_sx = pitch_image.shape[1]
        map_sy = pitch_image.shape[0]


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
        #fill vectors
        width_vec = []
        height_vec = []
        for ((x1, y1), (x2, y2)) in boxes:
            width_vec.append(abs(x1 - x2))
            height_vec.append(abs(y1 - y2))

        fh = get_field_of_view_horizontal(average_player_width(width_vec))
        fv = get_field_of_view_vertical(average_player_height(height_vec))

        fov_horizontal = ((frame_number - 1)*fov_horizontal + fh)/frame_number
        fov_vertical = ((frame_number - 1)*fov_vertical + fv)/frame_number

        scaled_width = get_scaled(fov_horizontal)
        scaled_height = get_scaled(fov_vertical)

        if len(lines) > 1:
            print('sta je')
            
            w = frame.shape[1]
            h = frame.shape[0]
            
            corner = corner_position(lines)
            s_corner = scaled_point(w, h, scaled_width, scaled_height, corner)

            center = (scaled_width/2, scaled_height/2)
            for (x,y) in positions:
                sx = scaled_point(w, h, scaled_width, scaled_height, (x,y))
                pos = player_pitch_position(center, s_corner, sx)
                if(pos[0] > 0 and pos[0] < 1 and pos[1] > 0 and pos[1] < 1):
                    pos=(int(pos[0]*map_sx), int(pos[1]*map_sy))
                    cv2.circle(pitch_image, pos, radius=5, color=(0, 0, 255), thickness=-1)
        
        concated = concat(frame, pitch_image, 'v')
        cv2.imshow('Match Detection', concated)

        frame_number += 1


if __name__ == "__main__":
    main()
