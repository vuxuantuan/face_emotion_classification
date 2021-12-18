import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import config
import utils

# create model for train
height = 48
width = 48
depth = 3
classes = 7

input_shape = (height, width, depth)
if K.image_data_format() == "channels_first":
    input_shape = (depth, height, width)

model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
# print(model.summary())
input_model = Input(shape=input_shape, name='image_input')
input_model = preprocess_input(input_model)
output_resnet_conv = model(input_model)
x = Dense(classes, activation='softmax', name='predictions')(output_resnet_conv)

model_resnet = Model(inputs=input_model, outputs=x)
print(model_resnet.summary())
plot_model(model_resnet, to_file="resnet50.png", show_shapes=True, show_layer_names=True, show_layer_activations=True,
           show_dtype=True)

"""load data from csv"""
train_data, train_label, val_data, val_label, test_data, test_label = utils.load_data_from_csv(config.data_fer)

if K.image_data_format() == "channels_first":
    train_data = train_data.reshape((train_data.shape[0], depth, height, width))
    val_data = val_data.reshape((val_data.shape[0], depth, height, width))
    test_data = test_data.reshape((test_data.shape[0], depth, height, width))
else:
    train_data = train_data.reshape((train_data.shape[0], height, width, depth))
    val_data = val_data.reshape((val_data.shape[0], height, width, depth))
    test_data = test_data.reshape((test_data.shape[0], height, width, depth))

# scale data to the range of [0, 1]
train_data = train_data.astype("float32") / 255.0
val_data = val_data.astype("float32") / 255.0
test_data = test_data.astype("float32") / 255.0

# convert the labels from integers to vectors
le = LabelBinarizer()
train_label = le.fit_transform(train_label)
val_label = le.fit_transform(val_label)
test_label = le.transform(test_label)

"""initialize the model and optimizer"""
epochs = 200
batch_size = 256
print("[INFO] compiling model...")
opt = utils.model_optims(epochs)
model_resnet.compile(optimizer=opt[2], loss="categorical_crossentropy", metrics=["accuracy"])

""" Training model"""
print("[INFO] training network...")
callbacks = [utils.model_early_stopping()]
augmentation = True
if augmentation:
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    # Train the networks with data augmentation
    H = model_resnet.fit(aug.flow(train_data, train_label, batch_size=batch_size),
                         validation_data=(val_data, val_label),
                         steps_per_epoch=len(train_data) // batch_size, epochs=epochs, verbose=1,
                         use_multiprocessing=True)
else:
    # train the network
    H = model_resnet.fit(train_data, train_label, validation_data=(val_data, val_label),
                         batch_size=batch_size, epochs=epochs, verbose=1,
                         use_multiprocessing=True)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model_resnet.predict(test_data, batch_size=batch_size)
print(classification_report(test_label.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=[str(x) for x in le.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, model_resnet.epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, model_resnet.epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, model_resnet.epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, model_resnet.epochs), H.history["val_accuracy"], label="val_acc")
plt.title("train Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(f"Loss_Accuracy_ResNet50_Augmentation_{augmentation}.png")
plt.show()
