import cv2
import matplotlib
matplotlib.use('macosx')
from matplotlib import pyplot as plt
from imageutils import image_with_gray_arg


def show_bgr_histogram(image, channels):
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("Flattened Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of pixels")
    for chan, color in zip(channels, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])


def show_2d_gb_histogram(figure, image, channels):
    ax = figure.add_subplot(131)
    hist = cv2.calcHist(
        [channels[1], channels[0]], [0, 1], None, [32, 32], [0, 256, 0, 256])
    p = ax.imshow(hist, interpolation="nearest")
    plt.colorbar(p)
    plt.xlabel("G")
    plt.ylabel("B")
    ax.set_title("GB Histogram")


def show_2d_gr_histogram(figure, image, channels):
    ax = figure.add_subplot(132)
    hist = cv2.calcHist(
        [channels[1], channels[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
    p = ax.imshow(hist, interpolation="nearest")
    plt.colorbar(p)
    plt.xlabel("G")
    plt.ylabel("R")
    ax.set_title("GR Histogram")


def show_2d_br_histogram(figure, image, channels):
    ax = figure.add_subplot(133)
    hist = cv2.calcHist(
        [channels[0], channels[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
    p = ax.imshow(hist, interpolation="nearest")
    plt.colorbar(p)
    plt.xlabel("B")
    plt.ylabel("R")
    ax.set_title("BR Histogram")
    print(f"BR 2D Histogram shape: {hist.shape}, {hist.flatten().shape[0]}")


image, gray_image = image_with_gray_arg()
channels = cv2.split(image)
cv2.imshow("Original", image)
show_bgr_histogram(image, channels)
fig = plt.figure()
show_2d_gb_histogram(fig, image, channels)
show_2d_gr_histogram(fig, image, channels)
show_2d_br_histogram(fig, image, channels)
plt.show()