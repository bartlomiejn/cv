import cv2
import matplotlib
matplotlib.use('macosx')
from matplotlib import pyplot as plt
from imageutils import image_with_gray_arg


def plot_histogram(histogram, name, y_label):
    plt.figure()
    plt.title("Grayscale Unnormalized Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of pixels")
    plt.plot(hist)
    plt.xlim([0, 256])


image, image_grayscale = image_with_gray_arg()
hist = cv2.calcHist([image_grayscale], [0], None, [256], [0, 256])
cv2.imshow("Grayscale", image_grayscale)
plot_histogram(hist, "Grayscale Unnormalized Histogram", "# of pixels")
hist /= hist.sum()
plot_histogram(hist, "Grayscale L1-Normalized Histogram", "% of pixels")
plt.show()