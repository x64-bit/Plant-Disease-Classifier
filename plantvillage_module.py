
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
img_rows, img_cols = 256, 256
input_shape = (img_rows, img_cols, 3)
# Training parameters
batch_size = 100
epochs = 25


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory="output/train",
    class_mode="categorical"
)

validation_generator = test_datagen.flow_from_directory("output/val")

print("input shape", input_shape)
# Block 1
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation=None, input_shape=input_shape))
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
model.add(Flatten())
# Decision layer
model.add(Dense(38, activation='softmax'))


# Compile and Train
model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
model.fit_generator(
    train_generator,
    steps_per_epoch=544,    # ceil(54309 imgs / 100 batch size)
    epochs=epochs, 
    validation_data=validation_generator)
model.evaluate_generator(generator=validation_generator, steps=50)
# print("loss:", score[0])
# print("acc:", score[1])
