from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import Model
import numpy as np
import pandas
import cv2
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model


# create model for train
height = 48
width = 48
depth = 3
input_shape = (height, width, depth)

if K.image_data_format() == "channels_first":
    input_shape = (depth, height, width)

model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
# print(model.summary())
input_model = Input(shape=input_shape, name='image_input')
input_model = preprocess_input(input_model)
output_resnet_conv = model(input_model)

x = Dense(7, activation='softmax', name='predictions')(output_resnet_conv)

model_resnet = Model(inputs=input_model, outputs=x)
print(model_resnet.summary())
plot_model(model_resnet, to_file="resnet50.png", show_shapes=True, show_layer_names=True, show_layer_activations=True,
           show_dtype=True)

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

    # convert image to 3 channels for pretrain vgg
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

if K.image_data_format() == "channels_first":
    train_data = train_data.reshape((train_data.shape[0], 3, 48, 48))
    val_data = val_data.reshape((val_data.shape[0], 3, 48, 48))
    test_data = test_data.reshape((test_data.shape[0], 3, 48, 48))
else:
    train_data = train_data.reshape((train_data.shape[0], 48, 48, 3))
    val_data = val_data.reshape((val_data.shape[0], 48, 48, 3))
    test_data = test_data.reshape((test_data.shape[0], 48, 48, 3))

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
epochs = 200
print("[INFO] compiling model...")
opt = SGD(learning_rate=0.01, decay=0.01 / epochs, momentum=0.9, nesterov=True)
model_resnet.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

print("[INFO] training network...")
augmentation = True
if augmentation:
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    # Train the networks with data augmentation
    H = model_resnet.fit_generator(aug.flow(train_data, train_label, batch_size=32),
                                   validation_data=(val_data, val_label),
                                   steps_per_epoch=len(train_data) // 32, epochs=epochs, verbose=1)
else:
    # train the network
    H = model_resnet.fit(train_data, train_label, validation_data=(val_data, val_label),
                         batch_size=32, epochs=epochs, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model_resnet.predict(test_data, batch_size=32)
print(classification_report(test_label.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=[str(x) for x in le.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
plt.title("train Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("Loss_ResNet50.png")
plt.show()
