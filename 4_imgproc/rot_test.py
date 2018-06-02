import argparse
import cv2


def parsed_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    return vars(ap.parse_args())


args = parsed_args()
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
print(image.shape)
(cX, cY) = (w / 2, h / 2)
rot_mat = cv2.getRotationMatrix2D((50, 50), 88, 1.0)
rotated = cv2.warpAffine(image, rot_mat, (w, h))
cv2.imshow("Rotated by 110 degrees counter-clockwise", rotated)
(b, g, r) = rotated[10, 10]
print("Pixel at (10, 10) R: {r}, G: {g}, B: {b}".format(r=r, g=g, b=b))
cv2.waitKey(0)
