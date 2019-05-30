from keras.layers import Input

from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Add

from keras.layers import MaxPool2D

from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout
from keras.layers import Dense

from keras.models import Model

def reduce_input(nb_filters):
    def f(x):
        if x.shape[-1] != nb_filters:
            x = Conv2D(nb_filters, kernel_size=(1, 1), padding='same')(x)
        return x
    return f


def residual_block_a(nb_filters, kernel_size=(3, 3)):
    """
        We are here defining a function in the residual_block function
        and we are returning this function so we can use it like a keras layer:
            `x = residual_block_a(64, kernel_size=(3, 3))(x)`
    """
    def f(x):
        orig = reduce_input(nb_filters)(x)

        conv = Conv2D(nb_filters, kernel_size=kernel_size, padding='same')(x)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

        conv = Conv2D(nb_filters, kernel_size=kernel_size, padding='same')(conv)
        conv = BatchNormalization()(conv)

        conv = Add()([conv, orig])

        conv = Activation('relu')(conv)
        return conv
    return f

def residual_block_b(nb_filters, kernel_size=(3, 3)):
    """
        We are here defining a function in the residual_block function
        and we are returning this function so we can use it like a keras layer:
            `x = residual_block_b(64, kernel_size=(3, 3))(x)`
    """
    def f(x):
        orig = reduce_input(nb_filters)(x)

        conv = Conv2D(nb_filters, kernel_size=kernel_size, padding='same')(x)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

        conv = Conv2D(nb_filters, kernel_size=kernel_size, padding='same')(conv)

        conv = Add()([conv, orig])

        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

        return conv
    return f

def residual_block_c(nb_filters, kernel_size=(3, 3)):
    """
        We are here defining a function in the residual_block function
        and we are returning this function so we can use it like a keras layer:
            `x = residual_block_c(64, kernel_size=(3, 3))(x)`
    """
    def f(x):
        orig = reduce_input(nb_filters)(x)

        conv = Conv2D(nb_filters, kernel_size=kernel_size, padding='same')(x)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

        conv = Conv2D(nb_filters, kernel_size=kernel_size, padding='same')(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

        conv = Add()([conv, orig])

        return conv
    return f

def residual_block_d(nb_filters, kernel_size=(3, 3)):
    """
        We are here defining a function in the residual_block function
        and we are returning this function so we can use it like a keras layer:
            `x = residual_block_d(64, kernel_size=(3, 3))(x)`
    """
    def f(x):
        orig = reduce_input(nb_filters)(x)

        conv = Activation('relu')(conv)
        conv = Conv2D(nb_filters, kernel_size=kernel_size, padding='same')(x)
        conv = BatchNormalization()(conv)

        conv = Activation('relu')(conv)
        conv = Conv2D(nb_filters, kernel_size=kernel_size, padding='same')(conv)
        conv = BatchNormalization()(conv)

        conv = Add()([conv, orig])

        return conv
    return f

def residual_block_e(nb_filters, kernel_size=(3, 3)):
    """
        We are here defining a function in the residual_block function
        and we are returning this function so we can use it like a keras layer:
            `x = residual_block_e(64, kernel_size=(3, 3))(x)`
    """
    def f(x):
        orig = reduce_input(nb_filters)(x)

        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Conv2D(nb_filters, kernel_size=kernel_size, padding='same')(x)

        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Conv2D(nb_filters, kernel_size=kernel_size, padding='same')(conv)

        conv = Add()([conv, orig])

        return conv
    return f

def TinyResNet(input_shape, nb_classes=10, block=None):
    """
        @return: Keras model based on the resnet Architecture
            `model = TinyResNet((100, 100, 3), 100)`
    """
    # Choose the residual block implementation to use
    if block == 'b' or block == residual_block_b:
        residual_block = residual_block_b
    elif block == 'c' or block == residual_block_c:
        residual_block = residual_block_c
    elif block == 'd' or block == residual_block_d:
        residual_block = residual_block_d
    elif block == 'e' or block == residual_block_e:
        residual_block = residual_block_e
    else:
        residual_block = residual_block_a

    # Model pipeline definition
    input = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(input)

    # Step 1
    x = residual_block(64, kernel_size=(3, 3))(x)
    x = residual_block(64, kernel_size=(3, 3))(x)

    x = MaxPool2D()(x)

    # Step 2
    x = residual_block(128, kernel_size=(3, 3))(x)
    x = residual_block(128, kernel_size=(3, 3))(x)

    x = MaxPool2D()(x)

    # Step 3
    x = residual_block(256, kernel_size=(3, 3))(x)
    x = residual_block(256, kernel_size=(3, 3))(x)

    x = MaxPool2D()(x)

    # Step 4
    x = residual_block(512, kernel_size=(3, 3))(x)
    x = residual_block(512, kernel_size=(3, 3))(x)

    # Final step
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    output = Dense(nb_classes, activation='softmax')(x)

    model = Model(input=input, output=output)

    return model

if __name__ == "__main__":
    input_shape = (100, 100, 3)
    model = TinyResNet(input_shape)

    model.summary()
