from imageutils import image_arg
from imageutils import wait_and_destroy_all_windows
import cv2


def compare_rgb(rgb_image):
    cv2.imshow("RGB", rgb_image)
    for (name, chan) in zip(("B", "G", "R"), cv2.split(rgb_image)):
        cv2.imshow(name, chan)
    wait_and_destroy_all_windows()


def compare_hsv(rgb_image):
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV", hsv_image)
    for (name, chan) in zip(("H", "S", "V"), cv2.split(hsv_image)):
        cv2.imshow(name, chan)
    wait_and_destroy_all_windows()


def compare_lab(rgb_image):
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2LAB)
    cv2.imshow("L*a*b*", lab_image)
    for (name, chan) in zip(("L*", "a*", "b*"), cv2.split(lab_image)):
        cv2.imshow(name, chan)
    wait_and_destroy_all_windows()


image = image_arg()
compare_rgb(image)
compare_hsv(image)
compare_lab(image)
