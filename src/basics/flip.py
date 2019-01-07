import imageutils
from imageutils import Axis
import cv2

image = imageutils.image_arg()
h_flipped = cv2.flip(image, flipCode=Axis.HORIZONTAL)
v_flipped = cv2.flip(image, flipCode=Axis.VERTICAL)
flipped = cv2.flip(image, flipCode=Axis.BOTH)
cv2.imshow("Original", image)
cv2.imshow("Horizontally flipped", h_flipped)
cv2.imshow("Vertically flipped", v_flipped)
cv2.imshow("Both axis", flipped)
cv2.waitKey(0)
