from __future__ import print_function
from descriptors import LocalBinaryPatterns
from imutils import paths
import numpy as np
import argparse
import cv2


def parsed_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d",
        "--dataset",
        required=True,
        help="Path to the dataset of shirt images"
    )
    ap.add_argument(
        "-q",
        "--query",
        required=True,
        help="Path to the query image"
    )
    return vars(ap.parse_args())


def chi_squared_distance(features, features2):
    return 0.5 * np.sum(((features - features2) ** 2)
                        / (features + features2 + 1e-10))


args = parsed_args()
descriptor = LocalBinaryPatterns(num_points=24, radius=8)
index = {}

for image_path in paths.list_images(args["dataset"]):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = descriptor.describe(gray)
    filename = image_path[image_path.rfind("/") + 1:]
    index[filename] = histogram

query = cv2.imread(args["query"])
query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
query_features = descriptor.describe(query_gray)

cv2.imshow("Query image", query)
results = {}

for k, features in index.items():
    results[k] = chi_squared_distance(features, query_features)

results = sorted([(v, k) for (k, v) in results.items()])[:3]

for i, (score, filename) in enumerate(results):
    print("#%d. %s: %.4f" % (i+1, filename, score))
    image = cv2.imread(args["dataset"] + "/" + filename)
    cv2.imshow("Result #{}".format(i+1), image)
    cv2.waitKey(0)
