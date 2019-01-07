import argparse
import cv2
import numpy as np


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument(
        "-l",
        "--lower-angle",
        type=float,
        default=90.0,
        help="Lower orientation angle")
    ap.add_argument(
        "-u",
        "--upper-angle",
        type=float,
        default=135.0,
        help="Upper orientation angle")
    return vars(ap.parse_args())


def sobel_gradient(image):
    # Image gradient - a directional change in image intensity.
    g_x = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0)
    g_y = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1)
    return g_x, g_y


def gradient_mag_orientation(g_x, g_y):
    # (0, 0) in the upper left corner
    # gradient_x = I(x+1, y) - I(x-1, y)
    # gradient_y = I(x, y+1) - I(x, y-1)
    magnitude = np.sqrt((g_x ** 2) + (g_y ** 2))
    orientation = np.arctan2(g_y, g_x) * (180 / np.pi) % 180
    return magnitude, orientation


args = args()
lower_angle = args["lower_angle"]
upper_angle = args["upper_angle"]
image = cv2.imread(args["image"])
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
g_x, g_y = sobel_gradient(image_gray)
mag, orientation = gradient_mag_orientation(g_x, g_y)
indices = np.where(orientation >= lower_angle, orientation, -1)
indices = np.where(orientation <= upper_angle, indices, -1)
mask = np.zeros(image_gray.shape, dtype="uint8")
mask[indices > -1] = 255
cv2.imshow("Masked", mask)
cv2.waitKey(0)