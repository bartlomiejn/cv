from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser


def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))


def predict(X, W):
    dot_product = X.dot(W)
    preds = sigmoid_activation(dot_product)
    # Threshold the outputs to binary class labels using a step function
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1
    return preds


ap = ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="Learning rate")
args = vars(ap.parse_args())