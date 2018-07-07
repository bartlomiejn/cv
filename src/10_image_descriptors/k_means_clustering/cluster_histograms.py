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


def generate_histograms(image_paths, descriptor):
    histograms = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        hist = descriptor.describe(image)
        histograms.append(hist)
    return histograms


args = get_args()
image_paths = get_image_paths()
histograms = generate_histograms(image_paths, LabHistogram(bins=[8, 8, 8]))
kmeans = KMeans(n_clusters=args["clusters"])
labels = kmeans.fit_predict(histograms)
print(f"Labels for input images: {labels}")
for label in np.unique(labels):
    label_paths = image_paths[np.where(labels == label)]
    for i, path in enumerate(label_paths):
        image = cv2.imread(path)
        cv2.imshow(f"Cluster {label+1}, Image {i+1}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
