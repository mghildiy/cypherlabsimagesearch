from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras import backend as K
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

class ShallowNet:

    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # network structure is CONV => RELU
        # Convolution layer has 32 filters(K), each of size 3 x 3
        # 'same' padding to ensure that output of convolution operation has same size as input
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))

        # we then apply ReLU activation
        model.add(Activation("relu"))

        # softmax classifier for final FC layer
        # flatten the multi dimensional representation
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model


