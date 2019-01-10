import sys
sys.path.insert(0, '..')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from support.datasets.datasetloader import DatasetLoader
from support.preprocessing.preprocessor import ResizePreprocessor
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to input dataset")
ap.add_argument(
    "-k",
    "--neighbours",
    type=int,
    default=1,
    help="# of nearest neighbours for classification"
)
ap.add_argument(
    "-j",
    "--jobs",
    type=int,
    default=-1,
    help="# of jobs for knn classifier (-1 uses all available cores)"
)
args = vars(ap.parse_args())

print(f"[INFO] Loading images from {args['dataset']}")
image_paths = list(paths.list_images(args["dataset"]))
dl = DatasetLoader(preprocessors=[ResizePreprocessor(32, 32)])
data, labels = dl.load(image_paths, verbose=500)

print(
    f"[INFO] data.shape: {data.shape}, "
    f"features matrix size: {data.nbytes / (1024 * 1000.0)}MB"
)
data = data.reshape((data.shape[0], 3072))

le = LabelEncoder()
labels = le.fit_transform(labels)
train_x, test_x, train_y, test_y = train_test_split(
    data,
    labels,
    test_size=0.25,
    random_state=42
)

print("[INFO] Evaluating the knn classifier")
model = KNeighborsClassifier(n_neighbors=args["neighbours"], n_jobs=args["jobs"])
model.fit(train_x, train_y)
report = classification_report(
    test_y,
    model.predict(test_x),
    target_names=le.classes_
)
print(report)