import cv2
import imutils


class LabHistogram:
    def __init__(self, bins):
        self.bins = bins

    def describe(self, image, mask=None):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        hist = cv2.calcHist(
            [lab], [0, 1, 2], mask, self.bins, [0, 256, 0, 256, 0, 256]
        )
        if imutils.is_cv2():
            normalized_hist = cv2.normalize(hist).flatten()
        else:
            # is_cv3()
            normalized_hist = cv2.normalize(hist, hist).flatten()
        return normalized_hist
