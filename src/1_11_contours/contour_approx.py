import cv2
from imageutils import image_with_gray_arg
from imageutils import Color
from contourutils import get_contours


image, image_gray = image_with_gray_arg()
contours = get_contours(image_gray, cv2.RETR_EXTERNAL)
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
    if len(approx) == 4: # If there are 4 vertices approximated, then its a rect
        cv2.drawContours(image, [contour], -1, Color.GREEN.value, 2)
        x, y, w, h = cv2.boundingRect(approx)
        cv2.putText(
            image, "Rectangle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            Color.GREEN.value, 2)
cv2.imshow("Rectangles", image)
cv2.waitKey(0)
