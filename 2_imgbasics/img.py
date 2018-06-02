import argparse
import cv2


def print_pixel(bgr_tuple):
    print("(0,0) R: {r}, G: {g}, B: {b}".format(r=bgr_tuple[0], g=bgr_tuple[1], b=bgr_tuple[2]))


def print_and_replace_first_pixel(image):
    (b, g, r) = image[0, 0]
    print_pixel((b, g, r))
    image[0, 0] = (0, 0, 255)
    (b, g, r) = image[0, 0]
    print_pixel((b, g, r))


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(parser.parse_args())

image = cv2.imread(args["image"])
print_and_replace_first_pixel(image)
(h, w) = image.shape[:2]

cv2.imshow("Original", image)
cv2.waitKey(0)
