import cv2
from imageutils import image_arg
from imageutils import wait_and_destroy_all_windows

image = image_arg()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def show_original():
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


def compare_morphological_op(op_type, elem_type):
    show_original()
    kernel_sizes = [(3, 3), (5, 5), (7, 7)]
    for size in kernel_sizes:
        kernel = cv2.getStructuringElement(elem_type, size)
        opening = cv2.morphologyEx(gray_image, op_type, kernel)
        op_title = morphological_op_name(op_type)
        description = "{}: ({}, {})".format(op_title, size[0], size[1])
        cv2.imshow(description, opening)
    wait_and_destroy_all_windows()


def compare_eroded():
    show_original()
    for i in range(0, 3):
        eroded = cv2.erode(gray_image.copy(), None, iterations=i + 1)
        cv2.imshow("Eroded {} times".format(i + 1), eroded)
    wait_and_destroy_all_windows()


def compare_dilated():
    show_original()
    for i in range(0, 3):
        eroded = cv2.dilate(gray_image.copy(), None, iterations=i + 1)
        cv2.imshow("Dilated {} times".format(i + 1), eroded)
    wait_and_destroy_all_windows()


def compare_opening():
    compare_morphological_op(cv2.MORPH_OPEN, cv2.MORPH_ELLIPSE)


def compare_closing():
    compare_morphological_op(cv2.MORPH_CLOSE, cv2.MORPH_ELLIPSE)


def compare_gradient():
    compare_morphological_op(cv2.MORPH_GRADIENT, cv2.MORPH_ELLIPSE)


compare_eroded()
compare_dilated()
compare_opening()
compare_closing()
compare_gradient()
