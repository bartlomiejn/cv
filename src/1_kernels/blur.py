import cv2
from imageutils import image_arg
from imageutils import wait_and_destroy_all_windows

img = image_arg()
kernel_sizes = [(3, 3), (9, 9), (15, 15)]


def show_original():
    cv2.imshow("Original", img)


def compare_average_blur():
    """Fast, but doesn't preserve edges"""
    show_original()
    for (kx, ky) in kernel_sizes:
        blurred_img = cv2.blur(img, (kx, ky))
        cv2.imshow("Average ({}, {})".format(kx, ky), blurred_img)
    wait_and_destroy_all_windows()


def compare_gaussian_blur():
    """Slightly slower and better at preserving edges"""
    show_original()
    for (kx, ky) in kernel_sizes:
        blurred_img = cv2.GaussianBlur(img, (kx, ky), sigmaX=0, sigmaY=0)
        cv2.imshow("Gaussian ({}, {})".format(kx, ky), blurred_img)
    wait_and_destroy_all_windows()


def compare_median_blur():
    """
    Used to reduce salt-and-pepper style noise as the median statistic is much
    less sensitive to outliers than other statistical methods like the mean
    """
    show_original()
    for k_size in (3, 9, 15):
        blurred_img = cv2.medianBlur(img, k_size)
        cv2.imshow("Median ({})".format(k_size), blurred_img)
    wait_and_destroy_all_windows()


def compare_bilateral_blur():
    """Substantially slower, removes detail while preserving edges"""
    show_original()
    params = [(11, 21, 7), (11, 41, 21), (11, 61, 39)]
    for (diameter, sigma_color, sigma_space) in params:
        blurred_img = cv2.bilateralFilter(
            img, diameter, sigma_color, sigma_space
        )
        title = "Bilateral d={} sc={} ss={}".format(
            diameter, sigma_color, sigma_space
        )
        cv2.imshow(title, blurred_img)
    wait_and_destroy_all_windows()


compare_average_blur()
compare_gaussian_blur()
compare_median_blur()
compare_bilateral_blur()
