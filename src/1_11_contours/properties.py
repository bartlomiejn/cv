import cv2
import numpy as np
from imageutils import wait_and_destroy_all_windows
from imageutils import image_with_gray_arg
from contourutils import get_contours

image, image_gray = image_with_gray_arg()
contours = get_contours(image_gray, cv2.RETR_EXTERNAL)
image_clone = image.copy()
for contour in contours:
    moments = cv2.moments(contour)
    c_x = int(moments["m10"] / moments["m00"])
    c_y = int(moments["m01"] / moments["m00"])
    cv2.circle(
        image_clone, center=(c_x, c_y), radius=10, color=(0, 255, 0),
        thickness=-1)
cv2.imshow("Original", image)
cv2.imshow("Centroids", image_clone)
cv2.waitKey(0)