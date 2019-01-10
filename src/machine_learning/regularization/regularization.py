import sys
sys.path.insert(0, '..')
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from support.preprocessing import ResizePreprocessor
from support.datasets import DatasetLoader
from imutils import paths
from argparse import ArgumentParser


def load_images(image_paths):
    dl = DatasetLoader(preprocessors=[ResizePreprocessor(32, 32)])
    data, labels = dl.load(image_paths, verbose=500)
    data = data.reshape((data.shape[0], 32 * 32 * 3))
    return data, labels


ap = ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to input dataset")
args = vars(ap.parse_args())

print("[INFO] Loading and preprocessing images")
image_paths = list(paths.list_images(args["dataset"]))
data, labels = load_images(image_paths)
le = LabelEncoder()
labels = le.fit_transform(labels)

train_X, test_X, train_y, test_y = train_test_split(
    data,
    labels,
    test_size=0.25,
    random_state=5
)

for r in None, "l1", "l2":
    print(f"[INFO] Training model with {r} penalty")
    model = SGDClassifier(
        loss="log",
        penalty=r,
        max_iter=10,
        learning_rate="constant",
        eta0=0.03,
        random_state=42
    )
    model.fit(train_X, train_y)
    acc = model.score(test_X, test_y)
    print(f"[INFO] {r} penalty accuracy: {(acc * 100):.2f}%")