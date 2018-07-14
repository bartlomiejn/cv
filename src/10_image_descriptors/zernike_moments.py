import cv2
import mahotas
import imutils
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


def box_points(box):
    return cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)


def draw_nonmatching_bboxes(image, smallest_idx, cnts):
    for idx, cnt in enumerate(cnts):
        if smallest_idx == idx:
            continue
        box = cv2.minAreaRect(cnt)
        box = np.int0(box_points(box))
        cv2.drawContours(image, [box], -1, (0, 0, 255), 2)


def draw_matching_bbox(image):
    pass


pattern = cv2.imread("../../assets/zernike_reference.jpg")
image = cv2.imread("../../assets/zernike_distractor.jpg")
_, pattern_features = describe_shapes(pattern)
cnts, image_features = describe_shapes(image)
dist = dist.cdist(pattern_features, image_features)
smallest_idx = np.argmin(dist)
draw_nonmatching_bboxes(image, smallest_idx, cnts)

box = cv2.minAreaRect(cnts[smallest_idx])
box = np.int0(box_points(box))
cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
x, y, w, h = cv2.boundingRect(cnts[smallest_idx])
cv2.putText(
    image, "PATTERN FOUND", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
    (0, 255, 0), 2
)

cv2.imshow("Image", image)
cv2.waitKey(0)
