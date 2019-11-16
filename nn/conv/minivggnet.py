from keras.models import Sequential
import keras.backend as K
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense

class MiniVGGNet:

    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        # variable for batch normalization
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # first CONV => RELU => CONV => RELU => POOL => DROPOUT
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        # CONV => RELU
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        # max pooling
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # dropout
        model.add(Dropout(0.25))

        # second CONV => RELU => CONV => RELU => POOL => DROPOUT
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        # CONV => RELU
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        # max pooling
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # dropout
        model.add(Dropout(0.25))

        # FC => RELU layer with 512 nodes
        # flatten the multi dimensional representation
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # final softmax classifier with number of nodes same as number of classes
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model


