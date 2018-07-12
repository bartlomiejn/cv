from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from argparse import ArgumentParser
import glob
import cv2
from contourutils import get_contours


def get_args():
    ap = ArgumentParser()
    ap.add_argument(
        "-d", "--dataset", required=True, help="Path to the dataset directory"
    )
    return vars(ap.parse_args())


args = get_args()
image_paths = sorted(glob.glob(args["dataset"] + "/*.jpg"))
data = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)[1]
    cnts = get_contours(thresh, cv2.RETR_EXTERNAL)
    biggest_cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest_cnt)
    roi = cv2.resize(thresh[y:y+h, x:x+w], (50, 50))
    moments = cv2.HuMoments(cv2.moments(roi)).flatten()
    data.append(moments)
distances = pairwise_distances(data).sum(axis=1)
i = np.argmax(distances)
image = cv2.imread(image_paths[i])
print(f"Found square: {image_paths[i]}")
cv2.imshow("Outlier", image)
cv2.waitKey(0)