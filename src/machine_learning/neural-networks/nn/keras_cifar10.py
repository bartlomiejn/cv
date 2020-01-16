from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras import backend as K
from keras.datasets.cifar import load_batch
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import platform


def load_cifar10():
    path = os.path.join(os.getcwd(), "../../../../datasets/cifar10")

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


ap = argparse.ArgumentParser()
ap.add_argument(
    "-o",
    "--output",
    required=True,
    help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

print("Loading CIFAR-10")

# There's this dumb SSL error on macOS that I don't have the patience to
# diagnose. In `datasets/cifar10` download the Python version of the CIFAR-10
# dataset, this will load it without doing any network requests.
if platform.system() == "Darwin":
    (train_x, train_y), (test_x, test_y) = load_cifar10()
else:
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()

train_x = train_x.astype("float") / 255.0
test_x = test_x.astype("float") / 255.0
train_x = train_x.reshape((train_x.shape[0], 3072))
test_x = test_x.reshape((test_x.shape[0], 3072))

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.fit_transform(test_y)

label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
    "horse", "ship", "truck"]

print("Training")

model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
    metrics=["accuracy"])

history = model.fit(train_x, train_y, validation_data=(test_x, test_y),
    epochs=100, batch_size=32)

print("Evaluating")

predictions = model.predict(test_x, batch_size=32)

print(classification_report(
    test_y.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=label_names))

print("Plotting the loss/accuracy")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), history.history["accuracy"],
    label="train_accuracy")
plt.plot(np.arange(0, 100), history.history["val_accuracy"],
    label="val_accuracy")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
