import sys; sys.path.insert(0, '.')
from support.nn.minivggnet import MiniVGGNet
from support.datasets.cifar10 import load_cifar10
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse
import platform


def step_decay(epoch):
    init_alpha = 0.01
    factor = 0.25
    drop_every = 5
    return float(init_alpha * (factor ** np.floor((1 + epoch) / drop_every)))


ap = argparse.ArgumentParser()
ap.add_argument(
    "-o",
    "--output",
    required=True,
    help="Output loss/accuracy plot filename")
ap.add_argument(
    "-m",
    "--modelname",
    required=True,
    help="Model file to save or load if the file already exists")
args = vars(ap.parse_args())

print("Loading CIFAR-10")

if platform.system() == "Darwin":
    (train_x, train_y), (test_x, test_y) = \
        load_cifar10("../../../../datasets/cifar10")
else:
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()

train_x = train_x.astype("float") / 255.0
test_x = test_x.astype("float") / 255.0

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
    "horse", "ship", "truck"]

print("Compiling model")

sgd = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
    metrics=["accuracy"])

print("Training network")

epoch_count = 40
callbacks = [LearningRateScheduler(step_decay)]

H = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=64,
    epochs=epoch_count, verbose=1, callbacks=callbacks)

print("Evaluating network")

predictions = model.predict(test_x, batch_size=64)

print(classification_report(
    test_y.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=label_names))

print("Plot history")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epoch_count), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epoch_count), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epoch_count), H.history["accuracy"],
    label="train_accuracy")
plt.plot(np.arange(0, epoch_count), H.history["val_accuracy"],
    label="val_accuracy")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])

print("Serialize model and save to file")

model_json = model.to_json()
with open(f"{args['modelname']}.json", "w") as file:
    file.write(model_json)
model.save_weights(f"{args['modelname']}.h5")
