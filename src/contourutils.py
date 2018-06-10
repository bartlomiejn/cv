import cv2
from imageutils import is_cv2


def contours_from_tuple(contours):
    return contours[0] if is_cv2() else contours[1]


def get_contours(image_gray, mode):
    contours = cv2.findContours(
        image_gray.copy(), mode, cv2.CHAIN_APPROX_SIMPLE)
    return contours_from_tuple(contours)