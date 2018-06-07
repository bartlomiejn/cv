import cv2
from imageutils import image_arg
from imageutils import wait_and_destroy_all_windows
from skimage.filters import threshold_local


def show_simple_thresholding(image, blurred_image):
    (T, thresh_inv) = cv2.threshold(
        blurred_image, thresh=200, maxval=255, type=cv2.THRESH_BINARY_INV)
    (T, thresh) = cv2.threshold(
        blurred_image, thresh=200, maxval=255, type=cv2.THRESH_BINARY)
    cv2.imshow("Original", image)
    cv2.imshow("Thresholded Inverse", thresh_inv)
    cv2.imshow("Thresholded", thresh)
    cv2.imshow("Masked", cv2.bitwise_and(image, image, mask=thresh_inv))
    wait_and_destroy_all_windows()


# - Otsu's thresholding is global
# - Otsu’s method assumes that our image contains two classes of pixels: the
# background and the foreground. Furthermore, Otsu’s method makes the
# assumption that the grayscale histogram of our pixel intensities of our
# image is bi-modal, which simply means that the histogram is two peaks.
# - Otsu’s method is a global thresholding method. In situations where lighting
# conditions are semi-stable and the objects we want to segment have sufficient
# contrast from the background, we might be able to get away with Otsu’s method.


def show_otsu_thresholding(image, blurred_image):
    (T, thresh_inv) = cv2.threshold(
        blurred_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cv2.imshow("Original", image)
    cv2.imshow(f"Otsu Thresholded T={T}", thresh_inv)
    cv2.imshow("Masked", cv2.bitwise_and(image, image, mask=thresh_inv))
    wait_and_destroy_all_windows()


# - Adaptive thresholding is local
# - For simple images with controlled lighting conditions, this usually isn’t a
# problem. But for situations when the lighting is non-uniform across the image,
# having only a single value of T can seriously hurt our thresholding
# performance.
# - In order to overcome this problem, we can use adaptive thresholding, which
# considers small neighbors of pixels and then finds an optimal threshold value
# T for each neighbor. This method allows us to handle cases where there may be
# dramatic ranges of pixel intensities and the optimal value of T may change for
# different parts of the image.
# - Choosing the size of the pixel neighborhood for local thresholding is
# absolutely crucial. The neighborhood must be large enough to cover sufficient
# background and foreground pixels.


def show_adaptive_thresholding(image, blurred_image):
    cv_thresh = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        blockSize=25, C=15)
    sk_t = threshold_local(
        blurred_image, block_size=29, offset=5, method="gaussian")
    # bitwise_not equivalent
    sk_thresh = (blurred_image < sk_t).astype("uint8") * 255
    cv2.imshow("OpenCV Mean Adaptive Thresholding", cv_thresh)
    cv2.imshow("Scikit Mean Adaptive Thresholding", sk_thresh)
    cv2.imshow("Original", image)
    cv2.waitKey(0)


image = image_arg()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Applying Gaussian blurring helps remove some of the high frequency edges in
# the image that we are not concerned with
blurred_image = cv2.GaussianBlur(gray_image, ksize=(7, 7), sigmaX=0)
show_simple_thresholding(image, blurred_image)
show_otsu_thresholding(image, blurred_image)
show_adaptive_thresholding(image, blurred_image)