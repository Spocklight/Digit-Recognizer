# %%
#Imports and config:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# %%
#Read the train and test datasets

print(os.getcwd())

train = pd.read_csv("/home/spocklight/Git/Git/Digit-Recognizer/data/train.csv")
test = pd.read_csv("/home/spocklight/Git/Git/Digit-Recognizer/data/test.csv")

# %%
#Pixel normalization:

train_data = train.drop(columns=['label']) / 255.0
train_labels = train['label']
test_data = test / 255.0

x_train , x_test , y_train , y_test = train_test_split(train_data, 
                                    train_labels, 
                                    test_size=0.1, 
                                    random_state=42)

# %%
#We have a 1D vector with 784 pixels and we have to reshape it to (28x28x1) before passing it to the CNN.

#first param in reshape is number of examples. We can pass -1 here as we want numpy to figure that out by itself

#reshape(examples, height, width, channels)
x_train = x_train.values.reshape(-1, 28, 28, 1)
x_test = x_test.values.reshape(-1, 28, 28, 1)
df_test= test_data.values.reshape(-1,28,28,1)

# %%
#Data Augmentation:

datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image 
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

#In this case, data augmentation with ImageDataGenerator does not directly modify the size of the training set x_train. 
#The transformation is applied in real-time when generating each batch during model training, 
#allowing the model to see a different version of each image in each epoch, but without duplicating or increasing the rows in the original dataset.

# %%
#One-hot-encoding:

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# %%
#We define the model:

# Steps:

# Use Sequential Keras API
# Add Convolutional Layers — Building blocks of ConvNets and what do the heavy computation
# Add Pooling Layers — Steps along image — reduces params and decreases likelihood of overfitting
# Add Batch Normalization Layer — Scales down outliers, and forces NN to not relying too much on a Particular Weight
# Add Dropout Layer — Regularization Technique that randomly drops a percentage of neurons to avoid overfitting (usually 20% — 50%)
# Add Flatten Layer — Flattens the input as a 1D vector
# Add Output Layer — Units equals number of classes. Sigmoid for Binary Classification, Softmax in case of Multi-Class Classification.
# Add Dense Layer — Fully connected layer which performs a linear operation on the layer’s input

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last',
                 input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid' ))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu', data_format='channels_last'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', strides=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# %%
