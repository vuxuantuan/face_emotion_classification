import config
import numpy as np
import pandas
import cv2
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import optimizers


def load_data_from_csv(name_csv=config.data_fer, pre_train_imagenet=True):
    data = pandas.read_csv(name_csv)

    train_data = []
    train_label = []
    val_data = []
    val_label = []
    test_data = []
    test_label = []
    for index, row in data.iterrows():
        emotion = row['emotion']
        pixels = row['pixels']
        usage = row['Usage']

        list_pixels = pixels.split()
        list_pixels = [int(pixel) for pixel in list_pixels]

        image = np.reshape(list_pixels, (48, 48))

        # convert image to 3 channels for pretrain vgg and resnet using ImageNet
        if pre_train_imagenet:
            image = cv2.merge([image, image, image])

        if usage == "Training":
            train_data.append(image)
            train_label.append(emotion)
        elif usage == "PrivateTest":
            val_data.append(image)
            val_label.append(emotion)
        elif usage == "PublicTest":
            test_data.append(image)
            test_label.append(emotion)

    # convert list to numpy array
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    val_data = np.array(val_data)
    val_label = np.array(val_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)

    return train_data, train_label, val_data, val_label, test_data, test_label


def model_early_stopping():
    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.00005, patience=11, verbose=1,
                                   restore_best_weights=True)
    return early_stopping


def model_lr():
    lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=7, min_lr=1e-7, verbose=1)
    return lr_scheduler


def model_callbacks():
    early_stopping = model_early_stopping()
    lr_scheduler = model_lr()
    callbacks = [early_stopping, lr_scheduler]
    return callbacks


def model_optims(epochs=120):
    optims = [
        optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam'),
        optimizers.Adam(0.001),
        optimizers.SGD(learning_rate=0.01, decay=0.01 / epochs, momentum=0.9, nesterov=True)
    ]
    return optims
