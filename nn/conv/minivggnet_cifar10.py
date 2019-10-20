import argparse
from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from nn.conv.minivggnet import MiniVGGNet
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# fetch cifar10 dataset from keras datasets. It has 60000 images/data points, each of 32 x 32 x 3 shape
# default split of 60000 and 10000
((trainX, trainY), (testX, testY)) = cifar10.load_data()

# scale the images to range [0, 1]. Also called as normalization
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# binarize/hot encode the labels
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# initialize and compile the model
print("[INFO] compiling model....")
lr = 0.01
numEpochs = 20
decay = lr / numEpochs
opt = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network....")
batchSize = 64
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=batchSize, epochs=numEpochs, verbose=1)

# evaluate the network
print("[INFO] evaluating network....")
predictions = model.predict(testX, batch_size=batchSize)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, numEpochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, numEpochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, numEpochs), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, numEpochs), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])