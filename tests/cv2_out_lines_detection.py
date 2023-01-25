import cv2
import numpy as np

video_path = '../videos/clip_1.mp4'
video = cv2.VideoCapture(video_path)


def line_intersect(line_A, line_B, segment=True):
    """ returns a (x, y) tuple or None if there is no intersection """
    ((Ax1, Ay1), (Ax2, Ay2)) = line_A
    ((Bx1, By1), (Bx2, By2)) = line_B
    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    if d:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
        uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
    else:
        return
    if segment and not (0 <= uA <= 1 and 0 <= uB <= 1):
        return
    x = Ax1 + uA * (Ax2 - Ax1)
    y = Ay1 + uA * (Ay2 - Ay1)

    return x, y


def get_lines(lines):
    new_lines = []
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
            x2 = x1 + 3 * (x2 - x1)
            y2 = y1 + 3 * (y2 - y1)
            new_lines.append(((x1, y1), (x2, y2)))
    return new_lines


def clip_lines(image, lines):
    new_lines = []
    xl = image.shape[1]-1
    yl = image.shape[0]-1
    for line in lines:
        inter1 = line_intersect(line, ((0, 0), (0, yl)))
        inter2 = line_intersect(line, ((0, 0), (xl, 0)))
        inter3 = line_intersect(line, ((xl, 0), (xl, yl)))
        inter4 = line_intersect(line, ((0, yl), (xl, yl)))

        if inter1 is not None:
            x1, y1 = inter1
            if inter2 is not None:
                x2, y2 = inter2
            elif inter3 is not None:
                x2, y2 = inter3
            elif inter4 is not None:
                x2, y2 = inter4
            else:
                raise Exception("Error occurred in line intersection.")
        elif inter2 is not None:
            x1, y1 = inter2
            if inter3 is not None:
                x2, y2 = inter3
            elif inter4 is not None:
                x2, y2 = inter4
            else:
                raise Exception("Error occurred in line intersection.")
        elif inter3 is not None:
            x1, y1 = inter3
            if inter4 is not None:
                x1, y1 = inter4
            else:
                raise Exception("Error occurred in line intersection.")
        else:
            raise Exception("Error occurred in line intersection.")

        new_lines.append(((int(x1), int(y1)), (int(x2), int(y2))))
    return new_lines


def resize(green, dim):
    resized = cv2.resize(green, dim, interpolation=cv2.INTER_AREA)
    return resized


def cumulative(green):
    new_green = np.ndarray(green.shape, dtype=int)
    new_green[0] = [int(bool(g)) for g in green[0]]
    for i in range(1, len(green)):
        new_green[i] = [int(bool(green[i, j])) + new_green[i-1, j] for j in range(len(green[i]))]
    return new_green


def fill_in_holes(green):
    contours, hierarchy = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(green, contours, -1, color=255, thickness=cv2.FILLED)
    green = 255-green
    contours, hierarchy = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(green, contours, -1, color=255, thickness=cv2.FILLED)
    green = 255 - green
    # cv2.imshow('Match Detection', green)
    return green


def get_top_lines(green, lines):
    xl = green.shape[1]-1
    yl = green.shape[0]-1
    green = fill_in_holes(green)
    green = resize(green, (200, 200))
    green = cumulative(green)
    g = sum(green[-1])

    def resize_line(line):
        ((x1, y1), (x2, y2)) = line
        new_line = ((x1 * 199 / xl, y1 * 199 / yl), (x2 * 199 / xl, y2 * 199 / yl))
        return new_line

    def get_zeros_above(line):
        new_line = resize_line(line)
        if new_line[1][0] == new_line[0][0]:
            return 0, 0
        if new_line[1][0] < new_line[0][0]:
            v = (new_line[0][0] - new_line[1][0]) * 200
            return v, v
        koef = (new_line[1][1] - new_line[0][1]) / (new_line[1][0] - new_line[0][0])
        gr = 0
        ngr = 0
        x1, x2 = int(new_line[0][0]), int(new_line[1][0] + 1)
        for x in range(min(x1, x2), max(x1, x2)):
            y = min(int(new_line[0][1] + koef * (x - new_line[0][0])), 199)
            gr += 200 - (y - green[y, x])
            ngr += green[y, x]
        return gr, ngr

    imsize = green.shape[0]*green.shape[1]
    best = (imsize**3, [])
    for i in range(len(lines)):
        for j in range(len(lines)):
            line1 = lines[i]
            line2 = lines[j]
            inter = line_intersect(line1, line2)
            if inter is not None:
                if line1[0][0] < inter[0] < line1[1][0]:
                    if line1[0][1] == 0 and line1[0][0] > line1[1][0]:
                        continue
                    ng4, g4 = 0, 0
                    if line1[0][1] == 0:
                        ng4, g4 = get_zeros_above(((0, 0), line1[0]))
                    ng1, g1 = get_zeros_above((line1[0], inter))
                    ng2, g2 = get_zeros_above((inter, line2[1]))
                    ng3, g3 = (0, 0) if line2[1][1] < yl else get_zeros_above((line2[1], (xl, yl)))
                    if line2[1][1] <= 0:
                        ng3, g3 = get_zeros_above((line2[1], (xl, 0)))
                    nga = ng1 + ng2 + ng3 + ng4
                    ga = g1 + g2 + g3 + g4
                    s = ga + nga
                    if s < best[0]:
                        inter = (int(inter[0]), int(inter[1]))
                        best = (s, [(line1[0], inter), (inter, line2[1])])
    for line in lines:
        if line[0][1] == 0 and line[0][0] > line[1][0]:
            continue
        ng3, g3 = 0, 0
        if line[0][1] == 0:
            ng3, g3 = get_zeros_above(((0, 0), line[0]))

        ng1, g1 = get_zeros_above(line)
        ng2, g2 = (0, 0) if line[1][1] < yl else get_zeros_above((line[1], (xl, yl)))
        if line[1][1] <= 0:
            ng2, g2 = get_zeros_above((line[1], (xl, 0)))

        nga = ng1 + ng2 + ng3
        ga = g1 + g2 + g3
        s = ga + nga
        if s < best[0]:
            best = (s, [line])
    return best[1]


def main():
    # Read the video frame by frame
    while video.isOpened():
        ret, image = video.read()
        if not ret:
            break
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([70, 255, 255])
        bitwise = cv2.bitwise_and(image, image, mask=cv2.inRange(hsv, lower_green, upper_green))
        edges = cv2.Canny(bitwise, 100, 200)

        green = cv2.inRange(hsv, lower_green, upper_green)
        green = fill_in_holes(green)
        # edges = cv2.Canny(green, 100, 200)

        lines = cv2.HoughLines(edges, 1, np.pi * 1 / 180, 150)
        lines = get_lines(lines)
        lines = clip_lines(image, lines)
        # image = cv2.cvtColor(green, cv2.COLOR_GRAY2RGB)
        # for ((x1, y1), (x2, y2)) in lines:
        #     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        lines = get_top_lines(green, lines)

        for ((x1, y1), (x2, y2)) in lines:
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # cv2.imshow('Match Detection', cumulative(resize(green, (200, 200))).astype(np.uint8))
        cv2.imshow('Match Detection', image)

    video.release()
    cv2.destroyAllWindows()

main()