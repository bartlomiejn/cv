from imageutils import image_with_gray_arg
import cv2

image, gray_image = image_with_gray_arg()
eq = cv2.equalizeHist(gray_image)
cv2.imshow("Original", gray_image)
cv2.imshow("Histogram equalization", eq)
cv2.waitKey(0)