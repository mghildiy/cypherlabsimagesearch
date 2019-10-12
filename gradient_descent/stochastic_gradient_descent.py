import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))


def predict(X, W):
    # take the dot product between our features and weight matrix
    predictions = sigmoid_activation(X.dot(W))

    predictions[predictions <= 0.5] = 0
    predictions[predictions > 0.5] = 1

    return predictions

def next_batch(X, y, batchSize):
    for i in np.arange(0, X.shape[0], batchSize):
        yield(X[i : i + batchSize], y[i : i + batchSize])


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
ap.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for SGD")
args = vars(ap.parse_args())

# generate dataset
# X is array of 1000 data points, with each data point a list of 2 features [x1 x2]
# y is a 1D array with 1000 elements [y1 y2.....y1000]
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
# reshape labels from a 1D array to 2D array with 1 column, 1000 rows
y = y.reshape((y.shape[0], 1))
# add 1 extra column to every data point so that weight matrix can hold biases too. X is now of shape [1000,3]
X = np.c_[X, np.ones((X.shape[0]))]

# prepare training and test data
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)


# weight matrix of shape [3 1]
W = np.random.randn(X.shape[1], 1)
losses = []

# train
for epoch in np.arange(0, args["epochs"]):
    epochLoss = []

    for (batchX, batchY) in next_batch(X, y, args["batch_size"]):
        predictions = sigmoid_activation(trainX.dot(W))
        # error is array of shape [1000 1]
        error = predictions - trainY
        loss = np.sum(error ** 2)
        epochLoss.append(loss)

        # gradient shape is [3 1]
        gradient = trainX.T.dot(error)
        W += -args["alpha"] * gradient

    loss = np.average(epochLoss)
    losses.append(loss)

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))

# evaluate our model
print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot the (testing) classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY.ravel(), s=30)

# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()