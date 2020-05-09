import argparse
import os
import platform
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
from support.nn.shallownet import ShallowNet
from support.datasets.cifar10 import load_cifar10


ap = argparse.ArgumentParser()
ap.add_argument(
    "-o",
    "--output",
    required=True,
    help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

print("Loading CIFAR-10")

(train_x, train_y), (test_x, test_y) = load_cifar10()

train_x = train_x.astype("float") / 255.0
test_x = test_x.astype("float") / 255.0

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
    "horse", "ship", "truck"]

print("Compiling model")

opt = SGD(lr=0.01)

model = ShallowNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

print("Training")

epoch_count = 40

H = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=32,
    epochs=epoch_count, verbose=1)

print("Evaluating")

predictions = model.predict(test_x, batch_size=32)
print(classification_report(
    test_y.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=label_names))

print("Plotting the loss/accuracy")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epoch_count), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epoch_count), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epoch_count), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, epoch_count), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
