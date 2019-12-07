import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
#from keras.applications import MobileNet
#from keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from keras.models import Sequential
from keras.optimizers import Adam



vgg16_model = keras.applications.vgg16.VGG16()
vgg16_model.summary()
type(vgg16_model)

'''model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)'''

model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

model.summary()
#model.layers.pop()
model.summary()

for layer in model.layers:
    layer.trainable = False

model.add(Dense(2, activation='softmax'))
model.summary()

train_path = 'data/train_path'

test_path = 'data/test_path'


train_batches = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input).flow_from_directory(
    train_path, target_size=(224,224), batch_size=10)

test_batches = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input).flow_from_directory(
    test_path, target_size=(224,224), batch_size=10,shuffle=False)

model.compile(Adam(lr=.001), loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(train_batches,steps_per_epoch=4,
                  epochs=5, verbose=2)

test_imgs, test_labels=next(test_batches)
#plot(test_imgs, titles=test_labels)

#test_labels= test_labels[:,0]

predictions = model.predict_generator(test_batches, steps=1, verbose=0)
print(predictions)

'''#Saving the  model to  use it later on
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json.dump(fer_json, json_file)
    #json_file.write(fer_json)
model.save_weights("fer.h5")'''

#Saving the  model to  use it later on
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")
