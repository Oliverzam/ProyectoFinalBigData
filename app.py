from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Cargar tu modelo entrenado
model = tf.keras.models.load_model('model/model_proyectofinal.h5')

# Diccionario de nombres de clases (modifica esto según tus clases)
class_names = {
    0: 'Red Dane cattle',
    1: 'Brown Swiss cattle',
    2: 'Ayrshire cattle',
    3: 'Jersey cattle',
    4: 'Holstein Friesian cattle'
}

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Ajusta el tamaño según tu modelo
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
                
            file.save(file_path)

            predictions = predict_image(file_path)
            result = np.argmax(predictions, axis=1)[0]
            class_name = class_names.get(result, "Unknown")
            return jsonify({'Predicted Class': class_name}), 200
        except Exception as e:
            app.logger.error(f"Failed to save file: {e}")
            return jsonify({'error': 'Failed to save file'}), 500

if __name__ == '__main__':
    app.run(debug=True)
