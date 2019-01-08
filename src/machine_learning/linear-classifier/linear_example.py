import numpy as np
import cv2

labels = ["dog", "cat", "panda"]
np.random.seed(1)

# Pseudorandom initialization of weight matrix and bias vector for this example
# with uniform sampling over the distribution [0, 1]
W = np.random.randn(3, 3072)
b = np.random.randn(3)

# Load "beagle.png", resize to (32, 32, 3) np.array, flatten to 3072-dim vector
original = cv2.imread("beagle.png")
image = cv2.resize(original, (32, 32)).flatten()

# Linear scoring function: dot product between the weight matrix and image
# pixels with bias added
scores = W.dot(image) + b

# Print out scores for each label and "predicted" class
for label, score in zip(labels, scores):
    print(f"[INFO] {label}: {score:.2f}")
cv2.putText(
    original,
    f"{labels[np.argmax(scores)]}",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 255, 0, 2)
)
cv2.imshow("Image", original)
cv2.waitKey(0)
