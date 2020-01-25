import sys; sys.path.insert(0, '../..')
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from support.preprocessing.preprocessor import ImageToArrayPreprocessor, ResizePreprocessor
from support.datasets.datasetloader import DatasetLoader
from support.nn.shallownet import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to input dataset")
ap.add_argument("-o", "--output", required=True, help="Path to output")
args = vars(ap.parse_args())

image_paths = list(paths.list_images(args["dataset"]))

rp = ResizePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

print("Loading dataset")

loader = DatasetLoader(preprocessors=[rp, iap])
(data, labels) = loader.load(image_paths, verbose=500)
data = data.astype("float") / 255.0

(train_x, test_x, train_y, test_y) = \
    train_test_split(data, labels, test_size=0.25, random_state=42)

train_y = LabelBinarizer().fit_transform(train_y)
test_y = LabelBinarizer().fit_transform(test_y)

print("Compiling")

sgd = SGD(lr=0.004)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
    metrics=["accuracy"])

print("Training")

H = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=32,
    epochs=100, verbose=1)

print("Evaluating")

pred = model.predict(test_x, batch_size=32)

print(classification_report(
    test_y.argmax(axis=1),
    pred.argmax(axis=1),
    target_names=["cat", "dog", "panda"]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])

print("Serialize model and save to file")

model_json = model.to_json()
with open("shallownet.json", "w") as file:
    file.write(model_json)
model.save_weights("shallownet.h5")

print("Finito!")