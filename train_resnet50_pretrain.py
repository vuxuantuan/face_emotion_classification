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

# create model for train
height = 48
width = 48
depth = 3
input_shape = (height, width, depth)

if K.image_data_format() == "channels_first":
    input_shape = (depth, height, width)

# model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
model = ResNet50(weights='imagenet')
print(model.summary())
