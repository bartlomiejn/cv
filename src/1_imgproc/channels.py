import numpy as np
import cv2
from imageutils import image_arg
from imageutils import print_pixel

img = image_arg()
(b_chan, g_chan, r_chan) = cv2.split(img)
merged_chans = cv2.merge([b_chan, g_chan, r_chan])
zeros = np.zeros(img.shape[:2], dtype="uint8")
cv2.imshow("Red", cv2.merge([zeros, zeros, r_chan]))
cv2.imshow("Green", cv2.merge([zeros, g_chan, zeros]))
cv2.imshow("Blue", cv2.merge([b_chan, zeros, zeros]))
cv2.imshow("Merged", merged_chans)
print_pixel(img[5, 80])
cv2.waitKey(0)
