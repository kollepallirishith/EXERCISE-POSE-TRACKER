from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained Keras model
model = tf.keras.models.load_model('model.h5')

# Define the class labels
class_labels = [
    'barbell biceps curl',
    'bench press',
    'chest fly machine',
    'deadlift',
    'hammer curl',
    'hip thrust',
    'lat pulldown',
    'lateral raises',
    'plank',
    'pull up',
    'push up',
    'shoulder press',
    'squat'
]

app = Flask(__name__)

def preprocess_image(image):
    # Resize and preprocess the image
    image = image.resize((224, 224))  # Resize to match model input
    img_array = np.array(image) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file.stream)

    # Preprocess the image
    img_array = preprocess_image(img)

    # Predict the class
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction, axis=1)[0]
    pest_name = class_labels[class_index]
    
    return jsonify({'pest_name': pest_name})

if __name__ == '__main__':
    app.run(debug=True)
