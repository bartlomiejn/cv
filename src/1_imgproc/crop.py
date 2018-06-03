import cv2

image = cv2.imread("../assets/florida_trip.png")
cropped_part = image[124:212, 225:380]
cv2.imshow("Cropped part", cropped_part)
cv2.waitKey(0)
