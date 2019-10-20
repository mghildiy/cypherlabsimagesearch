import sys
sys.path.append('../../')
from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from nn.conv.shallownet import ShallowNet
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# fetch cifar10 dataset from keras datasets. It has 60000 images/data points, each of 32 x 32 x 3 shape
# default split of 60000 and 10000
print("[INFO] loading CIFAR-10 data....")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)

# scale the images to range [0, 1]. Also called as normalization
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# binarize/hot encode the labels
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)
# trainY is now of shape (50000, 10)
print(trainY.shape)
# testY is now of shape (10000, 10)
print(testY.shape)
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# initialize and compile the model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = ShallowNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
num_epochs = 40
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=num_epochs, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, num_epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, num_epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, num_epochs), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, num_epochs), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()