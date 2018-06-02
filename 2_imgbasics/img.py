import argparse
import cv2


def print_pixel(rgb_tuple):
    print("(0,0) R: {r}, G: {g}, B: {b}".format(r=rgb_tuple[0], g=rgb_tuple[1], b=rgb_tuple[2]))


def print_first_pixel(image):
    (b, g, r) = image[0, 0]
    print_pixel((r, g, b))


def print_from_image(image, pixel_xy):
    print_pixel(image[pixel_xy[1], pixel_xy[0]])


def replace_first_pixel(image):
    image[0, 0] = (0, 0, 255)


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(parser.parse_args())

image = cv2.imread(args["image"])
print_first_pixel(image)
replace_first_pixel(image)
print_first_pixel(image)

print_from_image(image, (111, 225))

(h, w) = image.shape[:2]
(cX, cY) = (w // 2, h // 2)
topleft_image = image[0:cY, 0:cX]

image[0:cY, 0:cX] = (0, 255, 0)


cv2.imshow("Original", image)
cv2.imshow("Top-left slice", topleft_image)

cv2.waitKey(0)
