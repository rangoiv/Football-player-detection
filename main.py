import cv2
import imutils
import numpy as np
import argparse

HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def detect(frame):
    bounding_box_cordinates, weights = HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)

    for x, y, w, h in bounding_box_cordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Player', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.putText(frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow('output', frame)
    return frame

video_path = 'videos/clip_1.mp4'
video = cv2.VideoCapture(video_path)

# Check if camera opened successfully
if (video.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
while (video.isOpened()):
    # Capture frame-by-frame
    ret, frame = video.read()

    if not ret:
        break

    # Display the resulting frame
    detect(frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


# When everything done, release the video capture object
video.release()

# Closes all the frames
cv2.destroyAllWindows()