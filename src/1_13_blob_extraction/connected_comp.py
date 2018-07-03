from skimage.filters import threshold_local
from skimage import measure
import numpy as np
import cv2

image = cv2.imread("../../assets/license_plate.png")
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
v = cv2.split(hsv_image)[2]
t = threshold_local(v, 29, offset=15, method="gaussian")
thresh = (v < t).astype("uint8") * 255
cv2.imshow("License plate", image)
cv2.imshow("Thresholded plate", thresh)
labels = measure.label(thresh, neighbors=8, background=8)
mask = np.zeros(thresh.shape, dtype="uint8")
print(f"Found {len(np.unique(labels))} blobs")
cv2.waitKey(0)