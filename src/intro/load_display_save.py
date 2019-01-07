import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Path to the image")
parser.add_argument("-o", "--output", required=True, help="Output image name")
args = vars(parser.parse_args())

image = cv2.imread(args["image"])
print("width: %d px" % (image.shape[1]))
print("height: %d px" % (image.shape[0]))
print("channels: %d" % (image.shape[2]))

cv2.imshow("Image", image)
cv2.waitKey(0)

cv2.imwrite(args["output"], image)
