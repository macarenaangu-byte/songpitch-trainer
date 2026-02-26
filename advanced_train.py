import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. LOAD DATA
print("📂 Loading features...")
with open('audio_features.pkl', 'rb') as f:
    data = pickle.load(f)

X = np.array(data['features'])
y = np.array(data['labels'])

# 2. PREPARE LABELS
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
X = X[..., np.newaxis]

# Split data FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.15)

# 3. BUILD THE AUGMENTATION LAYER
# This adds noise inside the brain, so it doesn't take extra RAM!
data_augmentation = tf.keras.Sequential([
    layers.GaussianNoise(0.01), # Adds a tiny bit of "fuzz" to help the AI generalize
])

# 4. DEEPER MODEL STRUCTURE
model = models.Sequential([
    data_augmentation, # ⬅️ Augmentation happens here now!
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'), # Lowered from 256 to save RAM
    layers.Dropout(0.5),
    layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. THE NIGHT SHIFT
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("🚀 Starting the RAM-friendly Night Shift...")
# Using a smaller batch_size (16) also helps prevent the "Killed" error
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), 
          batch_size=16, callbacks=[early_stop])

model.save('song_pitch_model.h5')
print("✨ High-accuracy model saved!")