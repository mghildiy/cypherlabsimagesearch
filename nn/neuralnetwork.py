import numpy as np

class NeuralNetwork:

    def __init__(self, layers, alpha=0.1):
        # layers is list of integers representing number of nodes in each layer, eg. [2, 2, 1]
        # alpha is learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # we initialize weights for every i,j pair of layers till last two layers
        for i  in np.arange(0, len(layers)-2):
            # we create a weight matrix connecting every node in ith layer to every node in (i+!)th layer
            # we add another node to both layers for bias
            # so if there are m nodes in ith layer, n nodes in (i+!)th payer, we make a matrix of (m+1) x (n+1)
            w = np.random.randn(layers[i] + 1, layers[i+1] + 1)
            # we add matrix to list
            # also scale w by dividing by the square root of the number of nodes in the ith layer to normalize
            # the variance of each neuron’s output
            self.W.append(w / np.sqrt(layers[i]))

        # for last 2 layers, we do things a bit differently because we don't have bias node in last layer
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # construct and return a string representing network architecture
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    # sigmoid activation function
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        # assuming x has already been passed through sigmoid
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        # we first add 1 to each feature matrix/data point to enable bias training in weight matrix
        X = np.c_[X, np.ones(X.shape[0])]

        for epoch in np.arange(0, epochs):
            # loop over each data point and label
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))

    def fit_partial(self, x, y):
        # we maintain a list of output activations for each layer
        # data point is the first activation itself. So first entry is a 2d array of 1 row, column equals num of features
        A = [np.atleast_2d(x)]

        # we feed forward the input aak forward pass
        for layer in np.arange(0, len(self.W)):
            # for each pair of layers, first layer has M nodes, second layer has N nodes
            # activation is 1xM(output of first layer), weight matrix is MxN
            # dot product would give 1xN matrix, representing net to second layer
            net = A[layer].dot(self.W[layer])

            # then we apply activation function on net to get output of second layer, which in turn is input activation
            # for next pair of layer, and so on. Final entry in A is the output of the last layer in the network
            out = self.sigmoid(net)
            A.append(out)

        # backward pass aka backward propagation
        # first determine error: predicted - target
        error = A[-1] - y

        # we use chain rules ot determine deltas for all layers
        # for output layer, delta is simply the error for output layer times
        # derivative of activation function for output value
        D = [error * self.sigmoid_deriv(A[-1])]

        for layer in np.arange(len(A) - 2, 0, -1):
            # delta for current layer is calculated in 2 steps:
            # dot product of delta for last layer and  weight matrix of current layer
            delta = D[-1].dot(self.W[layer].T)
            # multiplying the delta with derivative of activation function for current layer
            delta = delta * self.sigmoid_deriv(A[layer])

            # update delta list
            D.append(delta)

        # update the weights
        # reverse the deltas
        D = D[::-1]

        for layer in np.arange(0, len(self.W)):
            # dot product of the current layer activation with the deltas of the current layer
            # and then multiplying with learning rate, finally adding to weight matrix of current layer
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        # forward the input across layers
        for layer in np.arange(0, len(self.W)):
            net = np.dot(p, self.W[layer])
            p = self.sigmoid(net)

        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss





