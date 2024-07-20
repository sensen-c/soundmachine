from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from combine_feature import generate_all_features_for_sound
from text_feature import convert_audio_to_text
from utils import convert_prediction_to_str
import logging
import pickle
import numpy as np
from vosk import Model



app = Flask(__name__)
app.logger.setLevel(logging.INFO)

UPLOAD_FOLDER = 'tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model_filename = './model/best_xgboost_model_without_early_stop.pkl'
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

tf_idf_filename = './model/tfidf_vectorizer.pkl'
with open(tf_idf_filename, 'rb') as file:
    tf_idf_model = pickle.load(file)

voice_to_text_model = Model("./model/vosk-model-en-us-0.22-lgraph")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'wavFile' not in request.files:
        app.logger.debug('No file part in the request')
        return jsonify({'error': 'No file part'}), 400

    file = request.files['wavFile']
    if file.filename == '':
        app.logger.debug('No selected file')
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.wav'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        result = predict_wav_file(file_path)

        return jsonify(result)

    app.logger.debug('Invalid file format')
    return jsonify({'error': 'Invalid file format'}), 400


def generate_text_features(file_path):
    app.logger.info(f"Starting text feature generation for file: {file_path}")
    processed_text = convert_audio_to_text(file_path,voice_to_text_model)
    app.logger.info("Successfully converted audio to text")

    try:
        tf_idf_vectors = tf_idf_model.transform(processed_text).toarray()
        app.logger.info(f"Generated TF-IDF vectors: {tf_idf_vectors}")
    except Exception as e:
        app.logger.error(f"Error generating TF-IDF vectors: {e}")
        raise e
    return tf_idf_vectors


def predict_wav_file(file_path):
    # Generate sound features for the sound file
    sound_features = generate_all_features_for_sound(file_path)
    app.logger.debug(f"Generated sound features: {sound_features}")

    # Transform the appropriate features using tf_idf_model
    tf_idf_features = generate_text_features(file_path)

    # Combine sound features and TF-IDF features
    all_features = [np.hstack((sound_features, tf_idf_features[0]))]
    app.logger.info(f"Combined features: {all_features}")

    # Predict using the loaded model
    model_prediction = loaded_model.predict(all_features)
    app.logger.info(f"Predictions are: {model_prediction}")

    # Convert the prediction to a string
    exposed_prediction = convert_prediction_to_str(model_prediction[0])
    return exposed_prediction

if __name__ == '__main__':
    app.run(debug=True)