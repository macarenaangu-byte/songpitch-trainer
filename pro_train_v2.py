import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight # This is the magic ingredient!

# 1. LOAD DATA
with open('audio_features.pkl', 'rb') as f:
    data = pickle.load(f)

X = np.array(data['features'])[..., np.newaxis]
X = np.repeat(X, 3, axis=-1)
X = tf.image.resize(X, [128, 128]).numpy()

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(data['labels'])

# --- NEW: CALCULATE CLASS WEIGHTS ---
# This tells the AI: "A mistake on Jazz is 5x more expensive than a mistake on Pop"
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
weight_dict = dict(enumerate(weights))

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.15)

# 2. LOAD & UNFREEZE (Full Brain)
model = models.load_model('song_pitch_model.h5')
for layer in model.layers:
    layer.trainable = True # Fully open for business

model.compile(optimizer=tf.keras.optimizers.Adam(1e-6), # Even SLOWER learning
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. TRAIN WITH WEIGHTS
print("🚀 Final Fine-Tuning: Forcing the AI to prioritize rare genres...")
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), 
          class_weight=weight_dict, callbacks=[callbacks.EarlyStopping(patience=5)])

model.save('song_pitch_model.h5')
print("✨ The Balanced Brain is ready!")