"""
Taken from:
https://kananvyas.medium.com/player-and-football-detection-using-opencv-python-in-fifa-match-6fd2e4e373f0
"""

# Import libraries
import cv2
import os
import numpy as np

# Reading the video
vidcap = cv2.VideoCapture('../videos/clip_2.mp4')
success, frame = vidcap.read()
count = 0
success = True
idx = 0

# Read the video frame by frame
while success:
    # converting into hsv image
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    k1 = 100
    k2 = 255
    lower_white = np.array([120, 150, 110])
    upper_white = np.array([255, 255, 255])

    # Define a mask ranging from lower to uppper
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Do masking
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Match Detection', res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    success, frame = vidcap.read()

vidcap.release()
cv2.destroyAllWindows()
