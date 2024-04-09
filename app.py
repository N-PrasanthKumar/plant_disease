from flask import Flask, render_template, request, jsonify, send_file
import os
import base64
from io import BytesIO
from tensorflow.python.keras.models import load_model
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import uuid
from PIL import Image 

model = load_model('trained_plant_disease_model.h5')
app = Flask(__name__)

dic = {
    0: 'Corn_(maize)_Common_rust',
    1: 'Tomato___Late_blight',
    2: 'Tomato___Leaf_Mold',
    3: 'Potato___Late_blight',
    4: 'Tomato___Tomato_mosaic_virus',
    5: 'Tomato___Spider_mites Two-spotted_spider_mite',
    6: 'Apple___healthy',
    7: 'Potato___healthy',
    8: 'Potato___Early_blight',
    9: 'Apple___Black_rot',
    10: 'Peach___healthy',
    11: 'Corn_(maize)_healthy',
    12: 'Pepper,bell__Bacterial_spot',
    13: 'Pepper,bell__healthy',
    14: 'Peach___Bacterial_spot',
    15: 'Tomato___Septoria_leaf_spot',
    16: 'Cherry_(including_sour)_healthy',
    17: 'Blueberry___healthy',
    18: 'Corn_(maize)_Northern_Leaf_Blight',
    19: 'Grape__Esca(Black_Measles)',
    20: 'Tomato___Early_blight',
    21: 'Strawberry___Leaf_scorch',
    22: 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
    23: 'Grape___Black_rot',
    24: 'Raspberry___healthy',
    25: 'Tomato___healthy',
    26: 'Apple___Cedar_apple_rust',
    27: 'Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot',
    28: 'Apple___Apple_scab',
    29: 'Orange__Haunglongbing(Citrus_greening)',
    30: 'Soybean___healthy',
    31: 'Tomato___Bacterial_spot',
    32: 'Grape___healthy',
    33: 'Cherry_(including_sour)_Powdery_mildew',
    34: 'Strawberry___healthy',
    35: 'Squash___Powdery_mildew',
    36: 'Tomato___Target_Spot',
    37: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
}

def predict_label(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) 
    img_array = np.array([img_array])
    predictions = model.predict(img_array)
    predicted_index = predictions.argmax(axis=-1)[0]
    return dic[predicted_index]

@app.route('/submit', methods=['POST'])
def get_output():
    if 'image' not in request.files:
        return render_template('index.html', error="No image selected")

    img = request.files['image']
    if img.filename == '':
        return render_template('index.html', error="No image selected")

    img_path = os.path.join('static/imgs/', str(uuid.uuid4()) + img.filename)
    img.save(img_path)
    
    try:
        prediction = predict_label(img_path)
        return render_template('index.html', prediction=prediction, img_path=img_path)
    except Exception as e:
        return render_template('index.html', error="Prediction failed: " + str(e))

@app.route('/save_photo', methods=['POST'])
def save_photo():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'})

    img_data = base64.b64decode(data['image'].split(',')[1])

    filename = str(uuid.uuid4()) + '.png'

    save_path = os.path.join('static/imgs/', filename)

    with open(save_path, 'wb') as f:
        f.write(img_data)

    try:
        prediction = predict_label(save_path)
        return jsonify({'success': True, 'prediction': prediction, 'filename': filename})
    except Exception as e:
        return jsonify({'error': "Prediction failed: " + str(e)})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
