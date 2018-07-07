from descriptors.labhistogram import LabHistogram
from sklearn.cluster import KMeans
from imutils import paths
import numpy as np
import argparse
import cv2


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d",
        "--dataset",
        required=True,
        help="Path to the input dataset directory"
    )
    ap.add_argument(
        "-k",
        "--clusters",
        type=int,
        default=2,
        help="# of clusters to generate"
    )
    return vars(ap.parse_args())


def get_image_paths():
    image_paths = list(paths.list_images(args["dataset"]))
    return np.array(sorted(image_paths))


args = get_args()
descriptor = LabHistogram([8, 8, 8])
histograms = []
for image_path in get_image_paths():
    image = cv2.imread(image_path)
    hist = descriptor.describe(image)
    histograms.append(hist)
kmeans = KMeans(n_clusters=args["clusters"])
labels = kmeans.fit_predict(histograms)