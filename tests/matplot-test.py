import matplotlib.pyplot as plt
import cv2


video_path = '../videos/clip_1.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap.release()

plt.imshow(frame[:,:,::-1]) # OpenCV uses BGR, whereas matplotlib uses RGB
plt.show()