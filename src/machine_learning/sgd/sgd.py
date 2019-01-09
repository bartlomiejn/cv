from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser


def parse_args():
    ap = ArgumentParser()
    ap.add_argument(
        "-e",
        "--epochs",
        type=float,
        default=100,
        help="# of epochs"
    )
    ap.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.01,
        help="Learning rate"
    )
    ap.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="Size of SGD mini-batches"
    )
    return vars(ap.parse_args())


def generate_2_class_classification_problem():
    X, y = make_blobs(
        n_samples=1000,
        n_features=2,
        centers=2,
        cluster_std=1.5,
        random_state=1
    )
    y = y.reshape((y.shape[0], 1))
    # Insert a column of 1s as the last entry in the feature matrix to treat the
    # bias as a trainable parameter
    X = np.c_[X, np.ones(X.shape[0])]
    return X, y


def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))


def predict(X, W):
    preds = sigmoid_activation(X.dot(W))
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1
    return preds


def next_batch(X, y, batch_size):
    for i in np.arange(0, X.shape[0], batch_size):
        yield (X[i:(i + batch_size)], y[i:(i + batch_size)])


def plot_classification(test_X, test_y):
    plt.style.use("ggplot")
    plt.figure()
    plt.title("data")
    plt.scatter(
        test_X[:, 0],
        test_X[:, 1],
        marker="o",
        c=test_y.ravel().tolist(),
        s=30
    )


def plot_losses(epochs_count, losses):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs_count), losses)
    plt.title("Training loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.show()


args = parse_args()
epochs_count = args["epochs"]
batch_size = args["batch_size"]
learning_rate = args["alpha"]
X, y = generate_2_class_classification_problem()
train_X, test_X, train_y, test_y = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=42
)
W = np.random.randn(X.shape[1], 1)
losses = []
print("[INFO] Training")
for epoch in np.arange(0, epochs_count):
    total_epoch_loss = []
    for batch_x, batch_y in next_batch(X, y, batch_size):
        preds = sigmoid_activation(batch_x.dot(W))
        error = preds - batch_y
        # Perform gradient descent update on a single batch rather than epoch
        gradient = batch_x.T.dot(error)
        W += -learning_rate * gradient
        # Update total epoch loss
        total_epoch_loss.append(np.sum(error ** 2))
    loss = np.average(total_epoch_loss)
    losses.append(loss)
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(f"[INFO] epoch={int(epoch + 1)}, loss={loss:.7f}")
print("[INFO] Evaluating")
preds = predict(test_X, W)
print(classification_report(test_y, preds))
plot_classification(test_X, test_y)
plot_losses(epochs_count, losses)
