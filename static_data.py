import pandas
import numpy as np
import cv2

data = pandas.read_csv('fer2013.csv')

# data_training = data.loc[data['Usage'] == 'PublicTest']
# print(data_training[['emotion', 'pixels']].groupby('emotion').count())

pixels = data.at[0, 'pixels']
list_pixels = pixels.split()
list_pixels = [int(pixel) for pixel in list_pixels]

image = np.reshape(list_pixels, (48, 48))

cv2.imwrite('fer2013_example.png', image)
