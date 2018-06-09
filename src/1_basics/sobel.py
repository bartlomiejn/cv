import cv2
from imageutils import image_arg
from imageutils import wait_and_destroy_all_windows
from enum import IntEnum


def show(function, image, image_gray):
    desc = None
    if function == cv2.Sobel:
        desc = "Sobel"
    elif function == cv2.Scharr:
        desc = "Scharr"
    g_x = function(image_gray, ddepth=cv2.CV_64F, dx=1, dy=0)
    g_y = function(image_gray, ddepth=cv2.CV_64F, dx=0, dy=1)
    g_x = cv2.convertScaleAbs(g_x)
    g_y = cv2.convertScaleAbs(g_y)
    image_sobel = cv2.addWeighted(g_x, 0.5, g_y, 0.5, 0)
    cv2.imshow("Original", image)
    cv2.imshow(f"{desc} X", g_x)
    cv2.imshow(f"{desc} Y", g_y)
    cv2.imshow(f"{desc} Combined", image_sobel)
    wait_and_destroy_all_windows()


image = image_arg()
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
show(cv2.Sobel, image, image_gray)
show(cv2.Scharr, image, image_gray)
