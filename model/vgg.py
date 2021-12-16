from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K

class VGGNet:
    @staticmethod
    def build(classes, height, width, depth):
        model = Sequential()
        input_shape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chanDim = 1

        model.add(Conv2D())