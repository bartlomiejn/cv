from skimage import feature
import numpy as np


class LocalBinaryPatterns:

    def __init__(self, num_points, radius):
        self.num_points = num_points
        self.radius = radius

    def describe(self, image, eps=1e-7):
        """
        Computes the LBP representation of the image and returns a histogram of
        patterns.
        """
        lbp = feature.local_binary_pattern(
            image,
            self.num_points,
            self.radius,
            method="uniform"
        )
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=range(0, self.num_points + 3),
            range=(0, self.num_points + 2)
        )
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        return hist
