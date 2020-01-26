import sys; sys.path.insert(0, '../..')

from support.nn.lenet import LeNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras import backend as K
from keras.optimizers import SGD
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument(
    "-o",
    "--output",
    required=True,
    help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

print("Loading MNIST")

mnist_path = os.path.join(
    os.getcwd(),
    "../../../../datasets/mnist/mnist-original.mat")
mnist_raw = loadmat(mnist_path)
dataset = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0],
    "COL_NAMES": ["label", "data"],
    "DESCR": "mldata.org dataset: mnist-original",
}

data = dataset["data"]

if K.image_data_format() == "channels_first":
    data = data.reshape(data.shape[0], 1, 28, 28)
else:
    data = data.reshape(data.shape[0], 28, 28, 1)

train_x, test_x, train_y, test_y = train_test_split(
    data / 255.0,
    dataset["target"].astype("int"),
    test_size=0.25,
    random_state=42)

lb = LabelBinarizer()

train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

print("Compiling model")

opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

print("Training network")

H = model.fit(train_x, train_y, validation_data=(test_x, test_y),
    batch_size=128, epochs=20, verbose=1)

print("Evaluating network")

predictions = model.predict(test_x, batch_size=128)

print(classification_report(
    test_y.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=[str(x) for x in lb.classes_]))

print("Plotting the loss/accuracy")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])