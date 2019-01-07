import cv2
import argparse
import numpy as np
from contourutils import get_contours


def args(ap):
    ap.add_argument(
        "-i", "--image", required=True, help="Path to the input image")
    ap.add_argument("-m", "--method", required=True, help="Sorting method")
    return vars(ap.parse_args())


def sort_contours(contours, method='left-to-right'):
    reverse = False
    i = 0
    if method == 'right-to-left' or method == 'bottom-to-top':
        reverse = True
    if method == 'top-to-bottom' or method == 'bottom-to-top':
        i = 1
    bboxes = [cv2.boundingRect(contour) for contour in contours]
    contours, bboxes = zip(*sorted(
        zip(contours, bboxes), key=lambda b: b[1][i], reverse=reverse))
    return contours, bboxes


def draw_contour(image, contour, i):
    moments = cv2.moments(contour)
    c_x = int(moments["m10"] / moments["m00"])
    c_y = int(moments["m01"] / moments["m00"])
    cv2.putText(
        image, f"#{i + 1}", (c_x - 20, c_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        (255, 255, 255), 2)
    return image


def get_edges(image):
    accum_edged = np.zeros(image.shape[:2], dtype="uint8")
    for channel in cv2.split(image):
        channel = cv2.medianBlur(channel, 11)
        edged = cv2.Canny(channel, 50, 200)
        accum_edged = cv2.bitwise_or(accum_edged, edged)
    return accum_edged


args = args(argparse.ArgumentParser())
image = cv2.imread(args["image"])
edge_map = get_edges(image)
contours = get_contours(edge_map, cv2.RETR_EXTERNAL)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
unsorted_contours = image.copy()
for (i, contour) in enumerate(contours):
    unsorted_contours = draw_contour(unsorted_contours, contour, i)
contours, bboxes = sort_contours(contours, method=args["method"])
for (i, contour) in enumerate(contours):
    draw_contour(image, contour, i)
cv2.imshow("Edge map", edge_map)
cv2.imshow("Unsorted contours", unsorted_contours)
cv2.imshow("Sorted contours", image)
cv2.waitKey(0)