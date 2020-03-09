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

# import the big boy model
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing.image import  ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# Model optimizer
optimizer = Adam(lr=0.001)
# Image dimms; MNIST is 28x28 pixels
img_rows, img_cols = 256, 256
input_shape = (img_rows, img_cols, 3)
# Training parameters
batch_size = 60
epochs = 100
seed = 69


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory="output/train",
    class_mode="categorical",
    seed=seed
)

validation_generator = test_datagen.flow_from_directory("output/val")

# Define MobileNetV2 model w/ ImageNet weights preloaded
base_model = MobileNetV2(weights='imagenet', include_top=False)

# Let's tweak the output a bit so that it works
x = base_model.output
# Apparently this flattens the conv layers but it works way better.
x = GlobalAveragePooling2D()(x)
# Extra fc layer because it's nice
# x = Dense(512, activation='relu')(x)
# Output layer (38 classes)
pred = Dense(38, activation='softmax')(x)

# Define model
model = Model(inputs=base_model.input, outputs=pred)

# Print out each layer
for i, layer in enumerate(model.layers):
  print(i, layer.name)

"""
# Freeze mobilenet layers so feature-extractors are preserved
for i, mobilenet_layer in enumerate(base_model.layers):
  mobilenet_layer.trainable = False
"""

# Compile model
model.compile(optimizer=optimizer, 
                loss="categorical_crossentropy",
                metrics=["accuracy"])

model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=epochs,
    validation_data=validation_generator)
model.evaluate_generator(generator=validation_generator, steps=50)
