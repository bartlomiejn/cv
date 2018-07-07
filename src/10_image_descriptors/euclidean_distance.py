from scipy.spatial import distance
from imutils import paths
import numpy as np
import cv2


def calculate_image_features(image_paths):
    image_features = {}
    for image_path in image_paths:
        image = cv2.imread(image_path)
        filename = image_path[image_path.rfind("/") + 1:]
        means, stdevs = cv2.meanStdDev(image)
        features = np.concatenate([means, stdevs]).flatten()
        image_features[filename] = features
    return image_features


image_paths = sorted(list(paths.list_images("../../assets/dinos")))
image_features = calculate_image_features(image_paths)
query_image = cv2.imread(image_paths[0])
cv2.imshow(f"Query {image_paths[0]}", query_image)
keys = sorted(image_features.keys())
for i, key in enumerate(keys):
    if key == "trex_01.png":
        continue
    image = cv2.imread(image_paths[i])
    dist = distance.euclidean(
        image_features["trex_01.png"], image_features[key]
    )
    cv2.putText(
        image, "%.2f" % dist, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        (0, 255, 0), 2
    )
    cv2.imshow(key, image)
cv2.waitKey(0)
