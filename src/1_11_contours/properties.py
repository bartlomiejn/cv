import cv2
import numpy as np
from enum import Enum
from imageutils import image_with_gray_arg
from imageutils import is_cv2
from contourutils import get_contours


class Color(Enum):
    WHITE = 255, 255, 255
    GREEN = 0, 255, 0


def area_and_perimeter(contour):
    return cv2.contourArea(contour), cv2.arcLength(contour, True)


def print_area_and_perimeter(index, contour):
    area, perimeter = area_and_perimeter(contour)
    print(f"Contour {index + 1}, area: {area}, perimeter: {perimeter:.2f}")


def get_centroid(contour):
    moments = cv2.moments(contour)
    c_x = int(moments["m10"] / moments["m00"])
    c_y = int(moments["m01"] / moments["m00"])
    return c_x, c_y


def draw_centroid(index, image, contour):
    c_x, c_y = get_centroid(contour)
    print(f"Centroid {index + 1}, c_x: {c_x}, c_y: {c_y}")
    cv2.circle(
        image, center=(c_x, c_y), radius=10, color=Color.GREEN.value,
        thickness=-1)
    cv2.putText(
        image, f"#{index + 1}", org=(c_x - 20, c_y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.25,
        color=Color.WHITE.value, thickness=1)


def draw_bbox(index, image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    print(f"Bbox {index + 1}, {x} {y} {w} {h}")
    cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h),
                  color=Color.GREEN.value, thickness=1)


def box_points(bbox):
    return cv2.cv.BoxPoints(bbox) if is_cv2() else cv2.boxPoints(bbox)


def draw_rotated_bbox(image, contour):
    bbox = cv2.minAreaRect(contour)
    bbox = np.int0(box_points(bbox))
    cv2.drawContours(image, [bbox], -1, Color.GREEN.value, 1)


def draw_min_enclosing_circle(index, image, contour):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    print(f"Circle {index + 1}, r: {radius}")
    cv2.circle(image, (int(x), int(y)), int(radius), Color.GREEN.value, 1)


def draw_ellipse(image, contour):
    # Has to have at least 5 points to fit an ellipse
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(image, ellipse, Color.GREEN.value, 1)


image, image_gray = image_with_gray_arg()
image_clone = image.copy()
contours = get_contours(image_gray, cv2.RETR_EXTERNAL)
for (idx, contour) in enumerate(contours):
    print_area_and_perimeter(idx, contour)
    draw_centroid(idx, image_clone, contour)
    draw_bbox(idx, image_clone, contour)
    draw_rotated_bbox(image_clone, contour)
    draw_min_enclosing_circle(idx, image_clone, contour)
    draw_ellipse(image_clone, contour)
cv2.imshow("Original", image)
cv2.imshow("Parameters", image_clone)
cv2.waitKey(0)