from flask import Flask, render_template
from flask import jsonify
from flask import request
import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from keras.models import load_model
from keras.models import model_from_json
import json
import base64
from io import BytesIO
import re
import os

import keras as ks

app = Flask(__name__)

def get_model():
    global model
    model = model_from_json(open("fer.json", "r").read())
    model.load_weights('fer.h5')
    print(" * Model loaded!")
    

def preprocess_image(image, target_size):
    if image.mode !="RGB":
        image = image.convert("RGB")
        image = image.resize(target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        return image
print(" * Loading keras model...")
get_model()


@app.route('/home')
def home():
    return render_template('predict.html')

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image,target_size=(224,224))
    
    prediction = model.predict_generator(processed_image,steps=1).tolist()
    
    response = {
        'prediction': {
            'cat': prediction[0][0],
            'dog': prediction[0][1]
            }
        }
    return jsonify(response)



''' base64_data = re.sub('^data:image/.+;base64,', '', str(message))# message should be in string format
    byte_data = base64.b64decode(base64_data + '=' * (-len(base64_data) % 4)) # padding error
    
    image_data= BytesIO(byte_data)
    
    image_data.seek(0)
    
    image = Image.open(image_data)'''
    
   
