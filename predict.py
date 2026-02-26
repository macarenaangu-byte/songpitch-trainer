import os
import pickle
import numpy as np
import tensorflow as tf
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

# 1. LOAD THE NEW 76% ACCURACY MODEL
model = tf.keras.models.load_model('song_pitch_model.h5')

# 2. REBUILD THE LABEL MAPPER FROM YOUR LATEST DATA
with open('audio_features.pkl', 'rb') as f:
    data = pickle.load(f)
encoder = LabelEncoder()
encoder.fit(data['labels']) 

print(f"✅ Specialist Ready! Detecting: {list(encoder.classes_)}")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file_path = "temp.mp3"
    file.save(file_path)

    try:
        # 3. AUDIO PROCESSING
        y, sr = librosa.load(file_path, sr=22050, duration=30)
        if len(y) < 22050 * 30:
            y = np.pad(y, (0, 22050 * 30 - len(y)))
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 4. DATA SHAPING
        X = mel_spec_db[np.newaxis, ..., np.newaxis]
        X = np.repeat(X, 3, axis=-1)
        X = tf.image.resize(X, [128, 128]).numpy()

        # 5. GENERATE PREDICTIONS
        preds = model.predict(X)[0]
        top_indices = preds.argsort()[-3:][::-1]
        
        # Convert NumPy strings to standard Python strings for the dashboard
        results = [str(encoder.inverse_transform([i])[0]) for i in top_indices]

        return jsonify({"predictions": results})

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=8000)