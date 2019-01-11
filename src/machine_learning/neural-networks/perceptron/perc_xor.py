import sys; sys.path.insert(0, '../..')
from support.nn.perceptron import Perceptron
import numpy as np

# XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

print("[INFO] Training")
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)

print("[INFO] Evaluating")
for x, target in zip(X, y):
    pred = p.predict(x)
    print(f"[INFO] data={x} ground-truth={target[0]} pred={pred}")