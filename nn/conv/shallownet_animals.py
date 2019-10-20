import sys
sys.path.append('../../')
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
import argparse
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from nn.conv.shallownet import ShallowNet
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# parse arguments to get images paths
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

# grab list of images
print("[INFO] loading images....")
imagePaths = list(paths.list_images(args["dataset"]))
# define preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()
# extract image data and labels
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

print(data.shape) #(3000, 32, 32, 3)
print(labels.shape) #(3000,)

# training data, test data split
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# hot-encode the labels
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)
print(trainY.shape) #(2250, 3)
print(testY.shape) #(750, 3)

# initialize and compile the network
print("[INFO] compiling the model....")
opt = SGD(lr=0.005)
model = ShallowNet.build(32, 32, 3, 3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training the model....")
num_epochs = 100
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=num_epochs, verbose=1)

# evaluate the network
print("[INFO] evaluating network....")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["cat", "dog", "panda"]))

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