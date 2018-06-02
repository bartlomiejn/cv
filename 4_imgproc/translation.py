import numpy as np
import argparse
import cv2


def parsed_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    return vars(ap.parse_args())


def translation_matrix(x, y):
    return np.float32([[1, 0, x], [0, 1, y]])


args = parsed_args()
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# | 1 0 tx |
# | 0 1 ty |

# Translate 25 in X, 50 in Y
M = translation_matrix(25, 50)
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow("Shifted down and right", shifted)

# Translate -50 in X, -90 in Y
M = translation_matrix(-50, -90)
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow("Shifted up and left", shifted)
cv2.waitKey(0)
