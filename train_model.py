import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# This is the "Gold Standard" way to import for TensorFlow 2.x
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

print(f"TensorFlow version: {tf.__version__}")

# 1. LOAD DATA
print("📂 Loading features...")
with open('audio_features.pkl', 'rb') as f:
    data = pickle.load(f)

X = np.array(data['features'])
y = np.array(data['labels'])

# 2. PREPARE DATA
# Encode labels (e.g., 'genre_Jazz' becomes 0, 'mood_Happy' becomes 1)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Add a 'channel' dimension (required for CNNs, like a color channel in a photo)
X = X[..., np.newaxis]

# Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

# 3. BUILD THE MODEL (The AI Brain Layers)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # ⬅️ New: Prevents memorization
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # ⬅️ New: Prevents memorization
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'), # Increased density
    layers.Dropout(0.5),   # ⬅️ New: Stronger dropout before output
    layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. TRAIN WITH STABILITY FEATURES
print("🧠 Training started with Early Stopping...")

# This stops training if the model stops improving for 5 rounds
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

# We increase epochs to 50 because the callback will stop it once it's "stable"
model.fit(
    X_train, 
    y_train, 
    epochs=50, 
    validation_data=(X_test, y_test), 
    batch_size=32,
    callbacks=[early_stop] # ⬅️ Crucial for stability
)

# 5. SAVE THE FINISHED BRAIN
model.save('song_pitch_model.h5')
print("✨ Stable model saved as 'song_pitch_model.h5'!")