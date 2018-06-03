import argparse
import cv2


def parsed_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    return vars(ap.parse_args())


def resized(image, height=None, width=None, interpolation=cv2.INTER_NEAREST):
    if width is None and height is None:
        return image
    dims = None
    if width is None:
        ratio = height / image.shape[0]
        dims = (int(image.shape[1] * ratio), height)
    else:
        ratio = width / image.shape[1]
        dims = (width, int(image.shape[0] * ratio))
    return cv2.resize(image, dims, interpolation)


args = parsed_args()
image = cv2.imread(args["image"])
inter_methods = [
    ("cv2.INTER_NEAREST", cv2.INTER_NEAREST),
    ("cv2.INTER_LINEAR", cv2.INTER_LINEAR),
    ("cv2.INTER_AREA", cv2.INTER_AREA),
    ("cv2.INTER_CUBIC", cv2.INTER_CUBIC),
    ("cv2.INTER_LANCZOS4", cv2.INTER_LANCZOS4)]
for (name, method) in inter_methods:
    resized_img = resized(image, width=int(image.shape[1]*1.5), interpolation=method)
    cv2.imshow("Interpolation method: {}".format(name), resized_img)
cv2.waitKey(0)
