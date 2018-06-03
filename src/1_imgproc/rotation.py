import imageutils
import cv2


def rotated(image, degrees, x_offset=None, y_offset=None):
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    if x_offset is int:
        cX += x_offset
    if y_offset is int:
        cY += y_offset
    rot_mat = cv2.getRotationMatrix2D((cX, cY), degrees, 1.0)
    return cv2.warpAffine(image, rot_mat, (w, h))


image = imageutils.arg_image()
cv2.imshow("Original", image)
cv2.imshow("Rotated by 45 degrees counter-clockwise", rotated(image, 45))
cv2.imshow("Rotated by 45 degrees c-c minus 50px offset on both axis", rotated(image, 45, x_offset=-50, y_offset=-50))
cv2.waitKey(0)
