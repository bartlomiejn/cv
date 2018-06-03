import imageutils
import numpy as np
import cv2

cv_max255 = cv2.add(np.uint8([200]), np.uint8([100]))
cv_min0 = cv2.subtract(np.uint8([50]), np.uint8([100]))
np_maxwrap = np.uint8([200]) + np.uint8([100])
np_minwrap = np.uint8([50]) - np.uint8([100])
print("cv2.add maxes out at 255: {}".format(cv_max255))
print("cv2.subtract mins out at 0: {}".format(cv_min0))
print("np + wraps around: {}".format(np_maxwrap))
print("np - wraps around: {}".format(np_minwrap))

image = imageutils.arg_image()
addition = np.ones(image.shape, dtype="uint8") * 75
subtraction = np.ones(image.shape, dtype="uint8") * 50
added_img = cv2.add(image, addition)
subtr_img = cv2.subtract(image, subtraction)
cv2.imshow("Original", image)
cv2.imshow("Increased intensity", added_img)
cv2.imshow("Decreased intensity", subtr_img)
imageutils.print_pixel(added_img[152, 61])
cv2.waitKey(0)
