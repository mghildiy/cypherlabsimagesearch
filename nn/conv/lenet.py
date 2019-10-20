from keras.models import Sequential
from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dense


class LeNet:

    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first set of CONV => RELU => POOL
        # we learn 20 filters
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL
        # we learn 50 filters
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # FC => RELU layer with 500 nodes
        # flatten the multi dimensional representation
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # final softmax classifier with number of nodes same as number of classes
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model

