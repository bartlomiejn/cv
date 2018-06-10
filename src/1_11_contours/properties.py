import cv2
from imageutils import image_with_gray_arg
from contourutils import get_contours
from enum import Enum


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


def draw_centroid(image, index, c_x, c_y):
    cv2.circle(
        image, center=(c_x, c_y), radius=10, color=Color.GREEN.value,
        thickness=-1)
    cv2.putText(
        image, f"#{idx + 1}", org=(c_x - 20, c_y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.25,
        color=Color.WHITE.value, thickness=1)


def draw_bounding_rect(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image_clone, pt1=(x, y), pt2=(x + w, y + h),
                  color=Color.GREEN.value, thickness=2)


image, image_gray = image_with_gray_arg()
image_clone = image.copy()
contours = get_contours(image_gray, cv2.RETR_EXTERNAL)
for (idx, contour) in enumerate(contours):
    print_area_and_perimeter(idx, contour)
    c_x, c_y = get_centroid(contour)
    draw_centroid(image_clone, idx, c_x, c_y)
    draw_bounding_rect(image_clone, contour)
cv2.imshow("Original", image)
cv2.imshow("Centroids", image_clone)
cv2.waitKey(0)