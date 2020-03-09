from tensorflow import keras

# MNIST
from keras.datasets import mnist

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



# Model optimizer
optimizer = Adam(lr=0.001)
# Image dimms; MNIST is 28x28 pixels
img_rows, img_cols = 28, 28
# Training parameters
batch_size = 60
epochs = 25




# Load mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape from (samples, length, width) to (samples, length, width, channels)
# to accommodate input
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
# Data conversion so NN doesn't bug out - research?
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# One-hot encode y_train and y_test
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)



# Block 1
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),
                 activation=None, input_shape=input_shape))
# Batch norm to make sure weights don't go haywire
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(strides=2))
model.add(Dropout(0.2))

# Block 2
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation=None))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(strides=2))
model.add(Dropout(0.2))

# Block 3
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation=None))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(strides=2))
model.add(Dropout(0.2))

# Flattening, but better: Instead of directly flattening,
# global max pooling first pools all feature maps together,
# then chugs into an FC layer
model.add(Flatten())
# Decision layer
model.add(Dense(10, activation='softmax'))


# Compile and Train
model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print("loss:", score[0])
print("acc:", score[1])
