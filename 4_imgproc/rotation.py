import argparse
import cv2


def parsed_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    return vars(ap.parse_args())


args = parsed_args()
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

(h, w) = image.shape[:2]
(cX, cY) = (w / 2, h / 2)

rot_mat = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
rotated = cv2.warpAffine(image, rot_mat, (w, h))
cv2.imshow("Rotated by 35 degrees", rotated)

rot_mat = cv2.getRotationMatrix2D((cX - 50, cY - 50), 45, 1.0)
rotated = cv2.warpAffine(image, rot_mat, (w, h))
cv2.imshow("Offset & rotated by 45 degrees", rotated)

cv2.waitKey(0)
