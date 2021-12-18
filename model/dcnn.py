from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend


class DCNN:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if backend.image_data_format() == "channels_first":
            input_shape = (depth, height, width)

        model = Sequential(name='DCNN')

        model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=input_shape, activation='elu', padding='same',
                         kernel_initializer='he_normal', name='conv2d_1'))

        model.add(BatchNormalization(name='BatchNormalization_1'))
        model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='elu', padding='same',
                         kernel_initializer='he_normal', name='conv2d_2'))

        model.add(BatchNormalization(name='BatchNormalization_2'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='MaxPooling2D_1'))
        model.add(Dropout(0.4, name='Dropout_1'))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='elu', padding='same',
                         kernel_initializer='he_normal', name='conv2d_3'))

        model.add(BatchNormalization(name='BatchNormalization_3'))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='elu', padding='same',
                         kernel_initializer='he_normal', name='conv2d_4'))

        model.add(BatchNormalization(name='BatchNormalization_4'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='MaxPooling2D_2'))
        model.add(Dropout(0.4, name='Dropout_2'))

        model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='elu', kernel_initializer='he_normal',
                         name='conv2d_5'))

        model.add(BatchNormalization(name='BatchNormalization_5'))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='elu', padding='same',
                         kernel_initializer='he_normal', name='conv2d_6'))

        model.add(BatchNormalization(name='BatchNormalization_6'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='MaxPooling2d_3'))
        model.add(Dropout(0.5, name='Dropout_3'))

        model.add(Flatten(name='Flatten'))

        model.add(Dense(128, activation='elu', kernel_initializer='he_normal', name='Dense_1'))

        model.add(BatchNormalization(name='BatchNormalization_7'))
        model.add(Dropout(0.6, name='Dropout_4'))
        model.add(Dense(classes, activation='softmax', name='Output_layer'))

        return model
