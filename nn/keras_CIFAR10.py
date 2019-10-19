from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# fetch cifar10 dataset from keras datasets. It has 60000 images/data points, each of 32 x 32 x 3 shape
# default split of 60000 and 10000
((trainX, trainY), (testX, testY)) = cifar10.load_data()
print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)

# scale the images to range [0, 1]. Also called as normalization
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
# flatten the datasets
trainX = trainX.reshape(trainX.shape[0], 32 * 32 * 3)
testX = testX.reshape(testX.shape[0], 32 * 32 * 3)
print(trainX.shape)
print(testX.shape)

# binarize/hot encode the labels
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)
# trainY is no of shape (50000, 10)
print(trainY.shape)
# testY is now of shape (10000, 10)
print(testY.shape)
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# define the 3072-1024-512-10 architecture using Keras
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

# train the network
print("[INFO] training network...")
# SGD optimizer with learning rate 0.01
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
numEpochs = 100
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=numEpochs, batch_size=32)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

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
#plt.savefig(args["output"])

print("End")