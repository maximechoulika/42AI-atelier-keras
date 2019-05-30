from keras.layers import Input

from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation

from keras.layers import MaxPool2D

from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout
from keras.layers import Dense

from keras.models import Model

def ConvBlock(nb_filters, kernel_size=(3, 3), padding='same', activation='relu'):
    """
        We are here defining a function in the ConvBlock function
        and we are returning this function so we can use it
        like a keras layer:
            `x = ConvBlock(64, kernel_size=(3, 3), padding='same')(x)`
    """
    def f(x):
        x = Conv2D(nb_filters, kernel_size=kernel_size, padding=padding)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x
    return f

def ModelBasic():

    input = Input(shape=input_shape)

    x = ConvBlock(32, kernel_size=(3, 3), padding='same')(input)
    x = MaxPool2D()(x)

    x = ConvBlock(64, kernel_size=(3, 3), padding='same')(x)
    x = ConvBlock(64, kernel_size=(3, 3), padding='same')(x)
    x = MaxPool2D()(x)

    x = ConvBlock(128, kernel_size=(3, 3), padding='same')(x)
    x = ConvBlock(128, kernel_size=(3, 3), padding='same')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    output = Dense(nb_classes, activation="softmax")(x)

    model = Model(input=input, output=output)

    return model
