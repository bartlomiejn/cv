from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser


def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))


def predict(X, W):
    preds = sigmoid_activation(X.dot(W))
    # Threshold the outputs to binary class labels using a step function
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1
    return preds


ap = ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="Learning rate")
args = vars(ap.parse_args())

# Generate a 2 class classification problem with 1000 data points, where each
# point is a 2-dim feature vector. Insert a column of 1s as the last entry in
# the feature matrix
X, y = make_blobs(
    n_samples=1000,
    n_features=2,
    centers=2,
    cluster_std=1.5,
    random_state=1
)
y = y.reshape((y.shape[0], 1))
X = np.c_[X, np.ones(X.shape[0])]

train_X, test_X, train_y, test_y = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=42
)


def train():



# Initialize the weight matrix with a uniform distribution and list of losses
W = np.random.randn(X.shape[1], 1)
losses = []

for epoch in np.arange(0, args["epochs"]):
    scores =  train_X.dot(W)
    preds = sigmoid_activation(scores)
    error = preds - train_y
    loss = np.sum(error ** 2)
    losses.append(loss)
    # Gradient descent update is the dot product between features and prediction
    # errors. We take a small step towards a set of more optimal parameters
    # based on the alpha parameter after each epoch
    gradient = train_X.T.dot(error)
    W += -args["alpha"] * gradient
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(f"[INFO] epoch={int(epoch + 1)}, loss={loss:.7f}")

print("[INFO] Evaluating")
preds = predict(test_X, W)
print(classification_report(test_y, preds))

# Plot the classification data and loss over time
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
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()