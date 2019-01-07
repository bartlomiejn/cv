import cv2
from contourutils import get_contours

image = cv2.imread("../../assets/more_shapes_example.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cnts = get_contours(image, cv2.RETR_EXTERNAL)
for idx, cnt in enumerate(cnts):
    x, y, w, h = cv2.boundingRect(cnt)
    roi = image[y:y+h, x:x+w]
    hu_moments = cv2.HuMoments(cv2.moments(roi)).flatten()
    print(f"Moments: {hu_moments}")
    cv2.imshow(f"ROI #{idx+1}", roi)

hu_moments = cv2.HuMoments(cv2.moments(image)).flatten()
print(f"[IMAGE] Moments: {hu_moments}")
cv2.waitKey(0)
