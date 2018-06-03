import numpy as np
import cv2
from imageutils import image_arg


def empty_canvas():
    return np.zeros(img.shape[:2], dtype="uint8")


def circle_mask():
    mask = empty_canvas()
    cv2.circle(mask, (145, 200), radius=100, color=255, thickness=-1)
    return mask


def rect_mask():
    mask = empty_canvas()
    cv2.rectangle(mask, (0, 90), (290, 450), color=255, thickness=-1)
    return mask


img = image_arg()
r_mask = rect_mask()
c_mask = circle_mask()
r_masked_img = cv2.bitwise_and(img, img, mask=r_mask)
c_masked_img = cv2.bitwise_and(img, img, mask=c_mask)
cv2.imshow("Original", img)
cv2.imshow("Mask", r_mask)
cv2.imshow("Rect masked image", r_masked_img)
cv2.imshow("Circle masked image", c_masked_img)
cv2.waitKey(0)
