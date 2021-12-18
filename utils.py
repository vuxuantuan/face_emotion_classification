import config
import numpy as np
import pandas
import cv2


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
