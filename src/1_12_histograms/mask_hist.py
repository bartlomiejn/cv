import cv2
import numpy as np
import matplotlib
matplotlib.use('macosx')
from matplotlib import pyplot as plt
from imageutils import image_arg


def plot_histogram(image, title, mask=None):
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    for chan, color in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])


image = image_arg()
cv2.imshow("Original", image)
plot_histogram(image, "Histogram for Original")
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (60, 290), (210, 390), 255, -1)
cv2.imshow("Mask", mask)
masked = cv2.bitwise_and(image, image, mask=mask)
plot_histogram(image, "Histogram for Masked", mask=mask)
cv2.imshow("Masked", masked)
plt.show()