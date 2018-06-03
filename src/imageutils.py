import argparse
import cv2
import numpy as np
from enum import IntEnum


class Axis(IntEnum):
    """Wraps `flipMode` values for `cv2.flip` function."""
    BOTH = -1
    VERTICAL = 0
    HORIZONTAL = 1


def parsed_args():
    """Parses required -i image path command-line argument."""
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    return vars(ap.parse_args())


def image_arg():
    """Attempts to retrieve image at `-i` command-line argument."""
    args = parsed_args()
    return cv2.imread(args["image"])


def print_pixel(bgr_pixel):
    """Prints BGR values from provided pixel."""
    (b, g, r) = bgr_pixel
    print("R: {r}, G: {g}, B: {b}".format(r=r, g=g, b=b))


def rotated(image, degrees, x_offset=None, y_offset=None):
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    if x_offset is int:
        cX += x_offset
    if y_offset is int:
        cY += y_offset
    rot_mat = cv2.getRotationMatrix2D((cX, cY), degrees, 1.0)
    return cv2.warpAffine(image, rot_mat, (w, h))


def translated(image, x, y):
    # Translation matrix:
    # | 1 0 tx |
    # | 0 1 ty |
    translation = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image, translation, (image.shape[1], image.shape[0]))
