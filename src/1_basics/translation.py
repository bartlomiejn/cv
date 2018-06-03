import cv2
import imageutils

image = imageutils.image_arg()
cv2.imshow("Original", image)
cv2.imshow("Shifted down and right", imageutils.translated(image, x=25, y=50))
cv2.imshow("Shifted up and left", imageutils.translated(image, x=-50, y=-90))
cv2.waitKey(0)
