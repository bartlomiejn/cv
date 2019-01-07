import cv2
from imageutils import image_with_gray_arg
from imageutils import Color
from contourutils import get_contours

# Aspect ratio = w / h, ratio < 1 means height is bigger than width
# Extent = shape area / bbox area
# Convex hull - Given a set of X points is the smallest possible convex set
# that contains X points
# Solidity = contour area / convex hull area

image, image_gray = image_with_gray_arg()
contours = get_contours(image_gray, cv2.RETR_EXTERNAL)
for idx, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    hull = cv2.convexHull(contour)
    hullArea = cv2.contourArea(hull)
    solidity = area / float(hullArea)
    char = None
    if solidity > 0.9:
        char = "O"
    elif solidity > 0.5:
        char = "X"
    if char:
        cv2.drawContours(image, [contour], -1, Color.GREEN.value, 3)
        cv2.putText(
            image, char, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
            (0, 255, 0), 4)
    print(f"{char} (Contour {idx + 1}) -- solidity = {solidity:.2f}")
cv2.imshow("Original", image)
cv2.waitKey(0)