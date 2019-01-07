import numpy as np
import cv2


def gen_canvas(x_size, y_size):
    return np.zeros((x_size, y_size, 3), dtype="uint8")


canvas = gen_canvas(300, 300)
green = (0, 255, 0)
red = (0, 0, 255)
blue = (255, 0, 0)
cv2.line(canvas, (0, 0), (300, 300), green)
cv2.line(canvas, (300, 0), (0, 300), red, 3)
cv2.rectangle(canvas, (10, 10), (60, 60), green)
cv2.rectangle(canvas, (50, 200), (200, 225), red, 5)
cv2.rectangle(canvas, (200, 50), (225, 125), blue, -1)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

canvas = gen_canvas(300, 300)
(cX, cY) = (canvas.shape[1] // 2, canvas.shape[0] // 2)
white = (255, 255, 255)
for r in range(0, 175, 25):
    cv2.circle(canvas, (cX, cY), r, white)
cv2.imshow("Circles", canvas)
cv2.waitKey(0)

canvas = gen_canvas(300, 300)
for i in range(0, 25):
    radius = np.random.randint(5, high=200)
    color = np.random.randint(0, high=256, size=(3,)).tolist()
    pt = np.random.randint(0, high=300, size=(2,))
    cv2.circle(canvas, tuple(pt), radius, color, -1)
cv2.imshow("Circles 2", canvas)
cv2.waitKey(0)
