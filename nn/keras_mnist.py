from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.preprocessing import LabelBinarizer

# parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# MNIST dataset
print("[INFO] loading MNIST (full) dataset...")
dataset = datasets.fetch_openml('mnist_784')
# data is numpy array of shape (70000, 784)
# normalise the data
data = dataset.data.astype("float") / 255.0
# target is numpy array of shape (70000,)
target = dataset.target
# split data for training and testing
(trainX, testX, trainY, testY) = train_test_split(data, target, test_size=0.25)
print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)

# binarize/hot encode the labels
lb = LabelBinarizer()
# hot-encode labels. Now each label is a vector label with 10 entries of 0,1.
# Only index represented by label is 1, others are 0.
# for eg, for an original label of  4, we now have vector [0 0 0 0 1 0 0 0 0 0]
# trainY now becomes numpy array of shape (52500, 10)
trainY = lb.fit_transform(trainY)
# testY now becomes numpy array of shape (17500, 10)
testY = lb.fit_transform(testY)

# now we define network architecture
# define the 784-256-128-10 architecture using Keras
model = Sequential()
# first layer has 784 nodes to be compatible with number of features in a data point(each image has 28x28=784 pixels)
# second layer has 256 nodes, and takes in an input with 784 features(same as the size of individual data point)
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
# third layer has 128 nodes, input size is decided by previous layer size
model.add(Dense(128, activation="sigmoid"))
# fourth layer has 10 nodes, same as number of entries in label vector, input size is decided by previous layer size
# we use softmax activation to obtain normalized class probabilities
model.add(Dense(10, activation="softmax"))

# train the network
print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
numEpochs = 100
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=numEpochs, batch_size=128)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
# predictions is an array of shape (17500, 10). Each row has 1o entries, with each entry being a probability value
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, numEpochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, numEpochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, numEpochs), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, numEpochs), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])

print("End")
