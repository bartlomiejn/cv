import numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.W = []
        self.layers = layers
        self.alpha = alpha

        for i in np.arange(0, len(layers) - 2):
            # Randomly initialize the weight matrix adding an extra node for the
            # bias
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)

            # w is scaled by dividing by the sqrt of the number of nodes in the
            # current layer, normalizing the variance of each neuron's output
            self.W.append(w / np.sqrt(layers[i]))

        # Last two layers are a case where the input connections need a bias
        # term, but the output does not
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        return f"NeuralNetwork: {'-'.join(str(1) for l in self.layers)}"

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        # Derivative of the sigmoid function assuming X already passed through
        # the sigmoid function
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, display_update_each=100):
        # Append a column of 1s for bias
        X = np.c_[X, np.ones((X.shape[0]))]
        for epoch in np.arange(0, epochs):

            # Train network through each individual data point
            for x, target in zip(X, y):
                self.fit_partial(x, target)

            # Display debug information
            if epoch == 0 or (epoch + 1) % display_update_each == 0:
                loss = self.calculate_loss(X, y)
                print(f"[INFO] epoch={epoch + 1}, loss={loss:.7f}")

    def fit_partial(self, x, y):
        # Construct a list of output activations for each layer as our data
        # point flows through the network
        A = [np.atleast_2d(x)]

        # Feedforward
        # Loop over the layers in the network
        for layer in np.arange(0, len(self.W)):
            # Get net input to the current layer by taking a dot product between
            # the activation and weight matrix
            net = A[layer].dot(self.W[layer])
            # Get net output by applying an activation function to the net input
            out = self.sigmoid(net)
            A.append(out)

        # Backpropagation
        # Compute the difference between our prediction (final output activation
        # in the activations list) and the target value
        # -1 means we want to access the last entry in the list in Python
        error = A[-1] - y

        # Build a list of deltas, first entry is the error of the output layer
        # multiplied by the derivative of the activation function for the output
        # value
        D = [error * self.sigmoid_deriv(A[-1])]

        # Chain rule implemented using a for loop
        for layer in np.arange(len(A) - 2, 0, -1):
            # Delta for the current layer is equal to:
            # 1) Delta of the previous layer dotted with the weight matrix
            # of the current layer
            # 2) Multiplying the delta by the derivative of the nonlinear
            # activation function for the activations of the current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        # Reverse the deltas
        D = D[::-1]

        # Weight update
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, add_bias=True):

        pred = np.atleast_2d(X)

        if add_bias:
            pred = np.c_[pred, np.ones((pred.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            pred = self.sigmoid(np.dot(pred, self.W[layer]))

        return pred

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, add_bias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss
