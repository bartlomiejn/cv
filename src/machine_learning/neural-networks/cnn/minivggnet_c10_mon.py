import argparse
import platform
import os
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from support.nn.minivggnet import MiniVGGNet
from support.datasets.cifar10 import load_cifar10
from support.logger.trainingmonitor import TrainingMonitor


ap = argparse.ArgumentParser()
ap.add_argument(
    "-o",
    "--output",
    required=True,
    help="Output loss/accuracy plot filename")
args = vars(ap.parse_args())

print(f"PID: {os.getpid()} Loading CIFAR-10")

(train_x, train_y), (test_x, test_y) = load_cifar10()

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
fig_path = os.path.join(os.getcwd(), args["output"], f"minivggnet_cifar10_{os.getpid()}.png")
json_path = os.path.join(os.getcwd(), args["output"], f"minivggnet_cifar10_{os.getpid()}.json")
print(f"Figure path: {fig_path}, json_path: {json_path}")
callbacks = [TrainingMonitor(fig_path, json_path=json_path)]

H = model.fit(train_x, train_y, validation_data=(test_x, test_y),
    batch_size=128, epochs=epoch_count, callbacks=callbacks, verbose=1)

print("Evaluating network")

predictions = model.predict(test_x, batch_size=64)

print(classification_report(
    test_y.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=label_names))

print("Serialize model and save to file")

model_json = model.to_json()
with open(f"{args['output']}.json", "w") as file:
    file.write(model_json)
model.save_weights(f"{args['output']}.h5")
