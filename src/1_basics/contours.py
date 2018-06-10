import cv2
from imageutils import wait_and_destroy_all_windows
from imageutils import image_with_gray_arg
from imageutils import is_cv2
import numpy as np


def contours_from_tuple(contours):
    return contours[0] if is_cv2() else contours[1]


def get_contours(image, mode):
    contours = cv2.findContours(image, mode, cv2.CHAIN_APPROX_SIMPLE)
    return contours_from_tuple(contours)


def show_all_contours(image, contours):
    clone = image.copy()
    cv2.drawContours(clone, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Original", image)
    cv2.imshow("Contours", clone)
    wait_and_destroy_all_windows()


def show_each_contour(image, contours):
    for (index, contour) in enumerate(contours):
        clone = image.copy()
        cv2.drawContours(clone, [contour], -1, (0, 255, 0), 2)
        cv2.imshow(f"Contour {index}", clone)
    wait_and_destroy_all_windows()


def show_external_contours(image, image_gray):
    clone = image.copy()
    contours = get_contours(image_gray.copy(), cv2.RETR_EXTERNAL)
    cv2.drawContours(clone, contours, -1, (0, 255, 0), 2)
    cv2.imshow("External contours", clone)
    wait_and_destroy_all_windows()


def show_contour_masking(image, image_gray):
    contours = get_contours(image_gray.copy(), cv2.RETR_EXTERNAL)
    for idx, contour in enumerate(contours):
        mask = np.zeros(image_gray.shape, dtype="uint8")
        cv2.drawContours(mask, [contour], -1, (255), -1)
        cv2.imshow("Original", image)
        cv2.imshow("Mask", mask)
        cv2.imshow(
            f"Masked contours: {idx}", cv2.bitwise_and(image, image, mask=mask))
        wait_and_destroy_all_windows()


image, image_gray = image_with_gray_arg()
contours = get_contours(image_gray.copy(), cv2.RETR_LIST)
show_all_contours(image, contours)
show_each_contour(image, contours)
show_external_contours(image, image_gray)
show_contour_masking(image, image_gray)
