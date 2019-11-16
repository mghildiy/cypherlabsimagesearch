import argparse
import os
from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from nn.conv.minivggnet import MiniVGGNet
from callbacks.trainingmonitor import TrainingMonitor
from pathlib import Path


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output directory for plot and json")
args = vars(ap.parse_args())

# show information on the process ID
print("[INFO process ID: {}".format(os.getpid()))

# load the training and testing data, then scale it into the range [0, 1]
((trainX, trainY), (testX, testY)) = cifar10.load_data()
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
numEpochs = 100
opt = SGD(lr=lr, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# construct the TrainingMonitor callback
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
data_folder = Path(args["output"])
figPath = data_folder / "{}.png".format(os.getpid())
jsonPath = data_folder / "{}.json".format(os.getpid())
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

# train the network
print("[INFO] training network....")
batchSize = 64
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=batchSize,
              epochs=numEpochs, callbacks=callbacks, verbose=1)