
import cv2


def concat(frame, pitch_image):
    height, width, _ = pitch_image.shape
    fheight, fwidth, _ = frame.shape

    fheight = int(fheight * width/fwidth)
    fwidth = int(width)
    resized = cv2.resize(frame, (fwidth, fheight), interpolation=cv2.INTER_AREA)
    concated = cv2.vconcat([resized, pitch_image])
    return concated


def main():
    video_path = '../videos/clip_1.mp4'
    pitch_image_path = '../images/pitch.png'

    video = cv2.VideoCapture(video_path)

    pitch_image = cv2.imread(pitch_image_path)

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        # Press q to exit video
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        concated = concat(frame, pitch_image)
        cv2.imshow('Match Detection', concated)

        # detections.render()  # Draws bounding boxes for detections on given image
        # cv2.imshow('output', detections.ims[0])


if __name__ == "__main__":
    main()
