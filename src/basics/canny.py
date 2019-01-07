import cv2
import numpy as np
from imageutils import image_arg


def median_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)


image = image_arg()
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(image_gray, (3, 3), sigmaX=0, sigmaY=0)
cv2.imshow("Original", image)
cv2.imshow("Blurred + Gray", blurred_image)
wide = cv2.Canny(blurred_image, 10, 200)
tight = cv2.Canny(blurred_image, 240, 250)
auto = median_canny(blurred_image)
cv2.imshow("Wide edge map", wide)
cv2.imshow("Tight edge map", tight)
cv2.imshow("Median-canny edge map", auto)
cv2.waitKey(0)
