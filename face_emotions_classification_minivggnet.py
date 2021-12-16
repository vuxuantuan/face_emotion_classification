from model import MiniVGGNet
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras import backend
import matplotlib.pyplot as plt
import numpy as np
import pandas

"""load data from csv"""
data = pandas.read_csv("fer2013.csv")

# convert data to image, label and split data to trainData, valData, testData
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

    if usage == "train":
        train_data.append(image)
        train_label.append(emotion)
    elif usage == "val":
        val_data.append(image)
        val_label.append(emotion)
    elif usage == "test":
        test_data.append(image)
        test_label.append(emotion)

# convert list to numpy array
train_data = np.array(train_data)
train_label = np.array(train_label)
val_data = np.array(val_data)
val_label = np.array(val_label)
test_data = np.array(test_data)
test_label = np.array(test_label)

if backend.image_data_format() == "channels_first":
    train_data = train_data.reshape((train_data.shape[0], 1, 48, 48))
    val_data = val_data.reshape((val_data.shape[0], 1, 48, 48))
    test_data = test_data.reshape((test_data.shape[0], 1, 48, 48))
else:
    train_data = train_data.reshape((train_data.shape[0], 48, 48, 1))
    val_data = val_data.reshape((val_data.shape[0], 48, 48, 1))
    test_data = test_data.reshape((test_data.shape[0], 48, 48, 1))

print(train_data[0].shape)

# scale data to the range of [0, 1]
train_data = train_data.astype("float32") / 255.0
val_data = val_data.astype("float32") / 255.0
test_data = test_data.astype("float32") / 255.0

# convert the labels from integers to vectors
le = LabelBinarizer()
train_label = le.fit_transform(train_label)
val_label = le.fit_transform(val_label)
test_label = le.transform(test_label)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01 / 120, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=48, height=48, depth=1, classes=7)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(train_data, train_label, validation_data=(val_data, val_label),
              batch_size=32, epochs=120, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(test_data, batch_size=32)
print(classification_report(test_label.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=[str(x) for x in le.classes_]))


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 120), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 120), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 120), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 120), H.history["val_accuracy"], label="val_acc")
plt.title("train Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()