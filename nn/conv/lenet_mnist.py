from sklearn import datasets
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from nn.conv.lenet import LeNet
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# MNIST dataset
print("[INFO] loading MNIST (full) dataset...")
dataset = datasets.fetch_openml('mnist_784')
data = dataset.data
labels = dataset.target
print(data.shape) # (70000, 784)
print(labels.shape) # (70000,)

# reshape the matrix based on ordering configured in keras.json
if K.image_data_format() == "channels_first":
    # num_samples x depth x rows x columns
    data = data.reshape(data.shape[0], 1, 28, 28)
else:
    # num_samples x rows x columns x depth
    data = data.reshape(data.shape[0], 28, 28, 1)


# train, test split
(trainX, testX, trainY, testY) = train_test_split(data/ 255.0, labels.astype("int"), test_size=0.25, random_state=42)

# binarize/hot-encode the labels
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)
print(trainY.shape) # (52500, 10)
print(testY.shape) # (17500, 10)

# initialize and compile the model
print("[INFO] compiling model....")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the model
print("[INFO] training the model....")
num_epochs = 1000
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, epochs=num_epochs, verbose=1)

# evaluate the network
print("[INFO] evaluating network....")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))

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
