import argparse
import cv2


def parsed_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    return vars(ap.parse_args())


args = parsed_args()
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

print(image.shape)

width = image.shape[1]
ratio = 150.0 / width
desired_dim = (150, int(image.shape[0] * ratio))

print(desired_dim)

resized = cv2.resize(image, desired_dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Resized to width", resized)

cv2.waitKey(0)
