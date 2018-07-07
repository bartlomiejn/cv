import cv2
from imageutils import Color
from contourutils import get_contours


def get_biggest_4pt_contour_approximation(contours):
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:7]
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        print(f"Original: {len(contour)}, approx: {len(approx)}")
        if len(approx) == 4:
            return approx


image = cv2.imread("../../assets/contours_receipt_original.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
edge_map = cv2.Canny(blurred, 75, 200)
contours = get_contours(edge_map.copy(), cv2.RETR_EXTERNAL)
contour = get_biggest_4pt_contour_approximation(contours)
contour_pts = contour.reshape(4, 2)
print(f"contour: {contour} contour_pts: {contour_pts}")
cv2.drawContours(image, [contour], -1, Color.GREEN.value, 2)
cv2.imshow("Edge map", edge_map)
cv2.imshow("Output", image)
cv2.waitKey(0)