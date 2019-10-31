# import the big boy model
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing.image import  ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# Define MobileNetV2 model w/ ImageNet weights preloaded
base_model = MobileNetV2(weights='imagenet', include_top=False)

# Let's tweak the output a bit so that it works
x = base_model.output
# Apparently this flattens the conv layers but it works way better. No time to research
x = GlobalAveragePooling2D()(x)
# Extra fc layer because it's nice
x = Dense(512, activation='relu')(x)
# Output layer: 9 diseases + 1 clean class = 10 classes
pred = Dense(3, activation='softmax')(x)

# Define model
model = Model(inputs=base_model.input, outputs=pred)

# Print out each layer
for i, layer in enumerate(model.layers):
  print(i, layer.name)


# We want to keep the conv-layers in MobileNet b/c they're what extract
# features - therefore we freeze MobileNet layers
for i, mobilenet_layer in enumerate(base_model.layers):
  mobilenet_layer.trainable = False

# Compile model
model.compile(optimizer="Adam", loss="categorical_crossentropy",
                metrics=["accuracy"])

# Define training data path for later
train_path = "train"
# Generate training images
train = ImageDataGenerator().flow_from_directory("C:/Users/Anjo P/Documents/hackathon/train", target_size=(256,256), classes=["Tomato___early_blight", "Tomato___healthy", "Tomato___Tomato_mosaic_virus"], batch_size=10)

# Fit model
model.fit(train, epochs=10)