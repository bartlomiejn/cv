import cv2
from imageutils import image_arg
from imageutils import wait_and_destroy_all_windows


def show_original(image):
    cv2.imshow("Original", image)


def morphological_op_name(op_type):
    if op_type == cv2.MORPH_OPEN:
        return "Opening"
    elif op_type == cv2.MORPH_CLOSE:
        return "Closing"
    elif op_type == cv2.MORPH_GRADIENT:
        return "Gradient"
    else:
        return "Other"


def function_name(function):
    if function == cv2.erode:
        return "Eroded"
    elif function == cv2.dilate:
        return "Dilated"
    else:
        return "Other"


def compare_morphological_op(op_type, elem_type, image, gray_image):
    show_original(image)
    kernel_sizes = [(3, 3), (5, 5), (7, 7)]
    for size in kernel_sizes:
        kernel = cv2.getStructuringElement(elem_type, size)
        opening = cv2.morphologyEx(gray_image, op_type, kernel)
        op_title = morphological_op_name(op_type)
        cv2.imshow(f"{op_title}: ({size[0]}, {size[1]})", opening)
    wait_and_destroy_all_windows()


def compare(function, image, gray_image):
    show_original(image)
    desc = function_name(function)
    for i in range(0, 3):
        eroded = function(gray_image.copy(), None, iterations=i + 1)
        cv2.imshow(f"{desc} {i + 1} times", eroded)
    wait_and_destroy_all_windows()


image = image_arg()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
compare(cv2.erode, image, gray_image)
compare(cv2.dilate, image, gray_image)
compare_morphological_op(cv2.MORPH_OPEN, cv2.MORPH_ELLIPSE, image, gray_image)
compare_morphological_op(cv2.MORPH_CLOSE, cv2.MORPH_ELLIPSE, image, gray_image)
compare_morphological_op(
    cv2.MORPH_GRADIENT, cv2.MORPH_ELLIPSE, image, gray_image)
