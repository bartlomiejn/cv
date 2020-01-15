import sys; sys.path.insert(0, '../..')
from support.nn.neuralnetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

print("Loading MNIST sample dataset")

digits = datasets.load_digits()

# Scale the pixel intensity values to the range [0, 1]
data = digits.data.astype("float")
data = (data - data.min()) / (data.max() - data.min())

print(f"Samples: {data.shape[0]}, dims: {data.shape[1]}")

(train_x, test_x, train_y, test_y) \
    = train_test_split(data, digits.target, test_size=0.25)

train_y = LabelBinarizer().fit_transform(train_y)
test_y = LabelBinarizer().fit_transform(test_y)

nn = NeuralNetwork([train_x.shape[1], 32, 16, 10])

print(f"{nn}")

nn.fit(train_x, train_y, epochs=1000)

predictions = nn.predict(test_x)
predictions = predictions.argmax(axis=1)
print(classification_report(test_y.argmax(axis=1), predictions))
