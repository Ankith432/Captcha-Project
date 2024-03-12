import base64
import io
from PIL import Image
import base64
import json

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
#from keras.preprocessing.image import ImageDataGenerator, image_to_array
#import numpy as np

from ctclayer_definition import decode_batch_predictions
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

#Loading Model
def get_model():
    global model
    model = load_model('captcha_ocr_model.keras')
    print(" * Model loaded! ")

def preprocess_image(encoded_image):
    image_height = 50
    image_width = 200

    try:

        # Decode base64 string to bytes
        decoded_bytes = base64.b64decode(encoded_image)
        # Convert bytes to TensorFlow tensor
        image = tf.io.decode_image(decoded_bytes, channels=3)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [image_height, image_width])
        image = tf.transpose(image, perm=[1, 0, 2])
        image = tf.expand_dims(image, axis=0)  # Add batch dimension

        return image

    except Exception as e:
        print(f"Error processing image: {e}")
        return None
    
print(" * Loading keras model...")
get_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        encoded_image = request.get_data(as_text=True)
        
        # Process the image
        processed_image = preprocess_image(encoded_image)

        if processed_image is None:
            response = {"error": "Failed to process the image"}
        else:
            # Make a prediction using the loaded model
            prediction_model = keras.models.Model(model.input[0], model.get_layer(name="dense2").output)
            prediction = prediction_model.predict(processed_image)

            # Decode the prediction
            pred_text = decode_batch_predictions(prediction)[0]

            response = {"prediction": pred_text}

            return jsonify(response)
    except Exception as e:
        print(f"Error: {str(e)}")
        return {'error': str(e)}

if __name__ == '__main__':
    app.run(debug=True)
