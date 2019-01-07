import cv2
import numpy as np
from imageutils import image_with_gray_arg
from contourutils import get_contours

def calculate_parameters(contour):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    extent = area / float(w * h)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / float(hull_area)
    return aspect_ratio, extent, hull, solidity


def shape_label(aspect_ratio, extent, solidity):
    shape = None
    if aspect_ratio >= 0.98 and aspect_ratio <= 1.02:
        shape = "SQUARE"
    elif aspect_ratio >= 3.0:
        shape = "RECTANGLE"
    elif extent < 0.65:
        shape = "L-PIECE"
    elif solidity > 0.8:
        shape = "Z-PIECE"
    return shape


image, image_gray = image_with_gray_arg()
thresh = cv2.threshold(image_gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
contours = get_contours(thresh, cv2.RETR_EXTERNAL)
hull_image = np.zeros(image_gray.shape[:2], dtype="uint8")
for index, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio, extent, hull, solidity = calculate_parameters(contour)
    shape = shape_label(aspect_ratio, extent, solidity)
    cv2.drawContours(hull_image, [hull], -1, 255, -1)
    cv2.drawContours(image, [contour], -1, (240, 0, 159), 3)
    cv2.putText(
        image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 0, 159), 2)
    print(f"Contour {index + 1} aspect_ratio:{aspect_ratio:.2f} "
          f"extent: {extent:.2f}, solidity: {solidity:.2f} {shape}")
cv2.imshow("Labeled pieces", image)
cv2.waitKey(0)