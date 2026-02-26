import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# 1. LOAD THE NEW BRAIN
print("🧠 Loading the Overnight Brain...")
model = tf.keras.models.load_model('song_pitch_model.h5')

# 2. LOAD DATA
print("📂 Loading features for comparison...")
with open('audio_features.pkl', 'rb') as f:
    data = pickle.load(f)

# Prep the audio images for the model
X = np.array(data['features'])[..., np.newaxis]
X = np.repeat(X, 3, axis=-1)
X = tf.image.resize(X, [128, 128]).numpy()

# Get the labels
encoder = LabelEncoder()
y_true = encoder.fit_transform(data['labels'])

# 3. RUN PREDICTIONS
print("📊 Analyzing the results (this might take a minute)...")
y_pred_probs = model.predict(X)
y_pred = np.argmax(y_pred_probs, axis=1)

# 4. PRINT THE TRUTH
print("\n" + "="*40)
print("🚀 FINAL GENRE ACCURACY REPORT")
print("="*40)
print(classification_report(y_true, y_pred, target_names=encoder.classes_))