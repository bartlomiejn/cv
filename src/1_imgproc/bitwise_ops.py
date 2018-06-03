import numpy as np
import cv2


def clear_canvas():
    return np.zeros((300, 300), dtype="uint8")


def rectangle_image():
    canvas = clear_canvas()
    return cv2.rectangle(canvas, (25, 25), (275, 275), color=255, thickness=-1)


def circle_image():
    canvas = clear_canvas()
    return cv2.circle(canvas, (150, 150), radius=150, color=255, thickness=-1)


rectangle = rectangle_image()
circle = circle_image()
bitwise_and = cv2.bitwise_and(rectangle, circle)
bitwise_or = cv2.bitwise_or(rectangle, circle)
bitwise_xor = cv2.bitwise_xor(rectangle, circle)
bitwise_not = cv2.bitwise_not(rectangle, circle)
cv2.imshow("AND", bitwise_and)
cv2.imshow("OR", bitwise_or)
cv2.imshow("XOR", bitwise_xor)
cv2.imshow("NOT", bitwise_not)
cv2.waitKey(0)