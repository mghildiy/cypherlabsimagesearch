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

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
args = vars(ap.parse_args())

# generate dataset
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
#print(X)
#print(y)
y = y.reshape((y.shape[0], 1))
X = np.c_[X, np.ones((X.shape[0]))]
#print(X)
#print(y)

# prepare training and test data
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)


#print("[INFO] training...")
# weight matrix: 3 rows, 1 column
W = np.random.randn(X.shape[1], 1)
#print(W)
losses = []

# train
#print("Training:", trainX)
print("Weights:", W)
#print("labels", trainY)
for epoch in np.arange(0, args["epochs"]):
    predictions = sigmoid_activation(trainX.dot(W))
    #print("predictions:", predictions)
    error = predictions - trainY
    #print("Error",error)
    loss = np.sum(error ** 2)
    #print("Loss",loss)
    losses.append(loss)

    gradient = trainX.T.dot(error)
    #print("gradient", gradient)
    W += -args["alpha"] * gradient
    #print("Updated weights", W)

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))


print("losses", losses)

# evaluate our model
print("[INFO] evaluating...")

print("Updated weights", W)
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot the (testing) classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
print(testY.shape)
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY.ravel(), s=30)

# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()