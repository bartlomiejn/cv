import sys; sys.path.insert(0, '../..')
from support.nn.neuralnetwork import NeuralNetwork
import numpy as np

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([2, 2, 1], alpha=0.5)
nn.fit(X, y, epochs=20000, display_update_each=1000)

for (x, target) in zip(X, y):
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print(f"[INFO] data={x}, ground_truth={target[0]}, pred={pred:.4f}, "
          f"step={step}")