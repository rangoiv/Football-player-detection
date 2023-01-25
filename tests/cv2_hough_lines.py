"""
Taken from:
https://kananvyas.medium.com/player-and-football-detection-using-opencv-python-in-fifa-match-6fd2e4e373f0
"""
import inspect
import collections
# Import libraries
import math

import cv2
import os
import numpy as np

# Reading the video
video_path = '../videos/clip_1.mp4'
video = cv2.VideoCapture(video_path)

# Read the video frame by frame
while video.isOpened():
    ret, image = video.read()

    if not ret:
        break

    # Press q to exit video
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    # converting into hsv image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # green range
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    # blue range
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Red range
    lower_red = np.array([0, 31, 255])
    upper_red = np.array([176, 255, 255])

    # white range
    lower_white = np.array([0, 0, 168])
    upper_white = np.array([172, 111, 255])

    green = cv2.bitwise_and(image, image, mask=cv2.inRange(hsv, lower_green, upper_green))

    white = cv2.bitwise_and(image, image, mask=cv2.inRange(hsv, lower_white, upper_white))

    # res_bgr = cv2.cvtColor(res1, cv2.COLOR_HSV2BGR)
    # res_gray = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
    #
    edges = cv2.Canny(green,100,200)
    # sobelxy = cv2.Sobel(src=white, ddepth=cv2.CV_64F, dx=12, dy=12, ksize=25)

    # cv2.imshow('Match Detection', res)
    # cv2.imshow('Match Detection', edges)

    lines = cv2.HoughLines(edges, 1, np.pi * 1 / 180, 100)
    print(len(lines))
    if lines is not None:
        for r_theta in lines:
            arr = np.array(r_theta[0], dtype=np.float64)
            r, theta = arr
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            x2 = x1 + 2*(x2-x1)
            y2 = y1 + 2 * (y2 - y1)
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=15)
    # if lines is not None:
    #     for l in lines:
    #         l = l[0]
    #         cv2.line(image, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    cv2.imshow('Match Detection', image)


video.release()
cv2.destroyAllWindows()
