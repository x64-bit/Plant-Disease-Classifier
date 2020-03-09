# Proper kfold splitter

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

# Layers
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPool2D

# Flattening, regularization, inter-layer data manipulation
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam

# Preprocessing
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.utils import to_categorical

import random

categories = ["Apple___Apple_scab", "Blueberry___healthy"]

training_data = []

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

datadir = "raw"

counter = 0
one_hot = OneHotEncoder(sparse=True)

for category in categories:
    counter = 0
    print(category)
    path = os.path.join(datadir, category)
    for img in os.listdir(path):
        if counter >= 10:
            break
        # print(img)
        # print(os.path.join(path, img))
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
        # new_array =  cv2.resize(img_array, (256,256))
        # plt.imshow(img_array)
        # plt.show()
        class_num = categories.index(category)
        training_data.append([img_array, class_num])
        counter += 1

# print(len(training_data))
# print(np.shape(training_data))
# print(np.zeros((20,2)))
# plt.imshow(training_data[0][0])
# plt.show()
# print(training_data[0])

kfold = StratifiedKFold(n_splits=2, shuffle=True)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

random.shuffle(X)
random.shuffle(y)

# plt.imshow(X[0])
# plt.show()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Block 1
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation=None, input_shape=(256,256,3)))
# Batch norm to make sure weights don't go haywire
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(strides=2))
model.add(Dropout(0.2))

# Block 2
model.add(Conv2D(32, kernel_size=(3, 3), activation=None))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(strides=2))
model.add(Dropout(0.2))

# Block 3
model.add(Conv2D(16, kernel_size=(3, 3), activation=None))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(strides=2))
model.add(Dropout(0.2))

# Flattening, but better: Instead of directly flattening,
# global max pooling first pools all feature maps together,
# then chugs into an FC layer
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
# Decision layer
model.add(Dense(2, activation='softmax'))

# model.summary()

# Compile and Train
model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("FUUUUUUUUUUUUUUUUCK")

for train_index, test_index in kfold.split(X, y):
    print(train_index)
    print(test_index)
    train_X, train_y, test_X, test_y = [], [], [], []
    # train_index is an array of INDEXES used to split
    for index in train_index:
        # plt.imshow(X[index])
        # plt.show()
        # debugging - print dataset array
        # print(X[index])
        # print(y[index])
        # generate temporary train dataset
        train_X.append(X[index])
        train_y.append(y[index])
        print("size: ", np.shape(train_y))
    for index in test_index:
        # print(X[index])
        # print(y[index])
        # generate temporary test dataset
        test_X.append(X[index])
        test_y.append(y[index])
    # make it np array so it's compatible
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)
    print(train_y)
    print(test_y)
    # one hot encode
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    print(train_y)
    print(test_y)
    print(np.shape(train_y))
    model.fit_generator(train_datagen.flow(train_X, train_y, batch_size=3),
                        steps_per_epoch=3, epochs=2)
    
    


# cool, now you have training data for each fold
# 
# now use ImageDataGenerator.flow for each fold to perform data augmentation and voila! you're done

"""
for train, test in kfold.split(X, y):
    print(np.shape)
    for sample in train:
        print(np.shape(sample))
        plt.imshow(sample)
        plt.show()
    for sample in test:
        print(sample)
"""
