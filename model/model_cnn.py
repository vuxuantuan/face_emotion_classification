from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K


def swish_activation(x):
    return (K.sigmoid(x) * x)


class CNN:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be "channels last"
        model = Sequential()
        input_shape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if backend.image_data_format() == " channels_first":
            input_shape = (depth, height, width)

        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same'))
        model.add(Conv2D(96, (3, 3), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64, activation=swish_activation))
        model.add(Dropout(0.4))
        model.add(Dense(7, activation='sigmoid'))

        return model

