import cv2
from imageutils import image_arg

image = image_arg()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def rect_kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_RECT, size)


def morphological_op_name(op_type):
    if op_type == cv2.MORPH_BLACKHAT:
        return "Blackhat"
    elif op_type == cv2.MORPH_TOPHAT:
        return "Tophat"
    else:
        return "Other"


def compare_morphological_op(op, elem):
    desc = "{}: {}x{}".format(morphological_op_name(op), elem[0], elem[1])
    cv2.imshow(desc, cv2.morphologyEx(gray_image, op, elem))


cv2.imshow("Original", image)
wide_elem = rect_kernel((13, 5))
# Blackhat operation finds dark regions on a light background
compare_morphological_op(cv2.MORPH_BLACKHAT, wide_elem)
# Tophat operation finds light regions on a dark background
compare_morphological_op(cv2.MORPH_TOPHAT, wide_elem)
cv2.waitKey(0)
