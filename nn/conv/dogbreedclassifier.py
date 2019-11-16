import sys
sys.path.append('../../')
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
import argparse
from datasets.simpledatasetloader import SimpleDatasetLoader
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from nn.conv.minivggnet import MiniVGGNet
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle


# parse arguments to get images paths
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-l", "--labels", required=True, help="path to labels csv")
args = vars(ap.parse_args())

# grab list of images
print("[INFO] loading images....")
imagePaths = list(paths.list_images(args["dataset"]))
labelPath = args["labels"]
# define preprocessors
width=32
height=32
sp = SimplePreprocessor(width, height)
iap = ImageToArrayPreprocessor()
# extract image data and labels
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.loadfromlabels(imagePaths, labelPath, verbose=500)

data, labels = shuffle(data, labels)

data = data.astype("float") / 255.0

print(data.shape)
print(labels.shape)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)

# hot-encode the labels
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
print(lb.classes_)

classStr = ''
for clas in lb.classes_:
	classStr += '&&' + clas

testY = lb.fit_transform(testY)
print(lb.classes_)
print(trainY.shape) #(7666, 120)
print(testY.shape) #(2556, 120)

#######################
aug = ImageDataGenerator(
		rotation_range=20,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest")
#######################

# initialize and compile the model
print("[INFO] compiling model....")
lr = 0.005
numEpochs = 100
decay = lr / numEpochs
opt = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
numClasses = 6
model = MiniVGGNet.build(width=width, height=height, depth=3, classes=numClasses)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network....")
batchSize = 64
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=batchSize, epochs=numEpochs, verbose=2)
#H = model.fit(aug.flow(trainX, trainY, batch_size=batchSize), validation_data=(testX, testY), batch_size=batchSize, epochs=numEpochs, verbose=2)

##########################
#H = model.fit_generator(
#	aug.flow(trainX, trainY, batch_size=batchSize),
#	validation_data=(testX, testY),
#	steps_per_epoch=len(trainX) // batchSize,
#	epochs=numEpochs,
#	verbose=2)
##########################

print("[INFO] serializing network...")
#model.save("C:\\work\\ml\\work\\cypherlabsimagesearch\\nn\\conv\\saved_models\\"+"dogbreeedclassifier_"+
#		   str(numEpochs)+"_"+str(numClasses)+"_all_weights.hdf5")
model.save("C:\\work\\ml\\work\\cypherlabsimagesearch\\nn\\conv\\saved_models\\"+"dogbreeedclassifier_"+
		   str(numEpochs)+classStr+"_weights.hdf5")

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, numEpochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, numEpochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, numEpochs), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, numEpochs), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Kaggle dog breed dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
#plt.show()
#plt.savefig("C:\work\ml\work\cypherlabsimagesearch\data\dogbreed\loss-accuracy-"+str(numEpochs)+
#			"_"+str(numClasses)+"_all_accuracy_loss.png")
plt.savefig("C:\work\ml\work\cypherlabsimagesearch\data\dogbreed\loss-accuracy-plots"+str(numEpochs)
			+classStr+"_accuracy_loss.png")