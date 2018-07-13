import cv2
import mahotas
import numpy as np
from scipy.spatial import distance as dist
from contourutils import get_contours


def describe_shapes(bgr_image):
    shape_features = []
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (13, 13), 0)
    thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)[1]
    thresholded = cv2.dilate(thresholded, None, iterations=4)
    thresholded = cv2.erode(thresholded, None, iterations=2)
    cnts = get_contours(thresholded, cv2.RETR_EXTERNAL)
    for cnt in cnts:
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        x, y, w, h = cv2.boundingRect(cnt)
        roi = mask[y:y+h, x:x+w]
        shape_features.append(mahotas.features.zernike_moments(
            roi,
            cv2.minEnclosingCircle(cnt)[1],
            degree=8
        ))
    return cnts, shape_features


pattern = cv2.imread("../../assets/zernike_reference.jpg")
image = cv2.imread("../../assets/zernike_distractor.jpg")
_, pattern_features = describe_shapes(pattern)
_, image_features = describe_shapes(image)
dist = dist.cdist(pattern_features, image_features)
i = np.argmin(dist)
print(f"pattern_features: {pattern_features}\nimage_features: {image_features}\ndistance: {dist}\nindex of pattern: {i}")