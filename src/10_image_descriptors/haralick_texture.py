from sklearn.svm import LinearSVC
from argparse import ArgumentParser
import mahotas
import glob
import cv2


def parsed_args():
    ap = ArgumentParser()
    ap.add_argument(
        "-d",
        "--training",
        required=True,
        help="Path to the training dataset"
    )
    ap.add_argument(
        "-t",
        "--testing",
        required=True,
        help="Path to the test dataset"
    )
    return vars(ap.parse_args())


def extract_haralick_features_and_labels_from(args):
    data = []
    labels = []
    for image_path in glob.glob(args["training"] + "/*.png"):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        texture = image_path[image_path.rfind("/") + 1:].split("_")[0]
        features = mahotas.features.haralick(image).mean(axis=0)
        data.append(features)
        labels.append(texture)
    return data, labels


def predict_images_from(args, model):
    for image_path in glob.glob(args["testing"] + "/*.png"):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = mahotas.features.haralick(gray).mean(axis=0)
        prediction = model.predict(features.reshape(1, -1))[0]
        cv2.putText(
            image,
            prediction,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3
        )
        cv2.imshow("Image", image)
        cv2.waitKey(0)


print("[INFO] Extracting features")
args = parsed_args()
data, labels = extract_haralick_features_and_labels_from(args)
print("[INFO] Training model")
model = LinearSVC(C=10.0, random_state=42)
model.fit(data, labels)
print("[INFO] Classifying images")
predict_images_from(args, model)