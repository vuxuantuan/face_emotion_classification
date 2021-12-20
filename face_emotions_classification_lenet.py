from model import LeNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras import backend
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import utils
import config

"""load data from csv"""
height = 48
width = 48
depth = 1
classes = 7
"""Load data from csv"""
train_data, train_label, val_data, val_label, test_data, test_label = utils.load_data_from_csv(config.data_fer, False)

if backend.image_data_format() == "channels_first":
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

# LeNet
# initialize the optimizer and model
print("[INFO] compiling model...")
model = LeNet.build(width=48, height=48, depth=1, classes=7)
epochs = 200
opt = utils.model_optims(epochs)
model.compile(loss="categorical_crossentropy", optimizer=opt[2], metrics=["accuracy"])

""" Training model"""
print("[INFO] training network...")
batch_size = 256
augmentation = False
if augmentation:
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    # Train the networks with data augmentation
    H = model.fit(aug.flow(train_data, train_label, batch_size=batch_size),
                  validation_data=(val_data, val_label),
                  steps_per_epoch=len(train_data) // batch_size, epochs=epochs, verbose=1, use_multiprocessing=True)
else:
    # train the network
    H = model.fit(train_data, train_label, validation_data=(val_data, val_label),
                  batch_size=batch_size, epochs=epochs, verbose=1, use_multiprocessing=True)

""" Evaluate the network """
print("[INFO] evaluating network...")
predictions = model.predict(test_data, batch_size=batch_size)
print(classification_report(test_label.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=[str(x) for x in le.classes_]))

""" Plot the training loss and accuracy """
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(H.history["loss"])), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, len(H.history['val_loss'])), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, len(H.history['accuracy'])), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, len(H.history['val_accuracy'])), H.history["val_accuracy"], label="val_acc")
plt.title("train Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(f"Loss_Accuracy_LeNet_Augmentation_{augmentation}.png")
plt.show()
