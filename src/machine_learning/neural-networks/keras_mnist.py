from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.io import loadmat
import os

ap = argparse.ArgumentParser()
ap.add_argument(
    "-o",
    "--output",
    required=True,
    help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

print("Loading MNIST dataset")

mnist_path = os.path.join(os.getcwd(), "../../../datasets/mnist/mnist-original.mat")

mnist_raw = loadmat(mnist_path)
dataset = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0],
    "COL_NAMES": ["label", "data"],
    "DESCR": "mldata.org dataset: mnist-original",
}

data = dataset["data"].astype("float") / 255.0
train_x, test_x, train_y, test_y \
    = train_test_split(data, dataset["target"], test_size=0.25)

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

sgd = SGD(0.01)

model.compile(
    loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

print("Training network")

H = model.fit(
    train_x,
    train_y,
    validation_data=(test_x, test_y),
    epochs=100,
    batch_size=128)

print("Evaluating network")

predictions = model.predict(test_x, batch_size=128)

print(classification_report(
    test_y.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=[str(x) for x in lb.classes_]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])