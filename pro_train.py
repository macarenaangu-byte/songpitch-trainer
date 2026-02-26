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

# 2. PREPARE DATA FOR TRANSFER LEARNING
# Pre-trained models expect 3 channels (RGB). We will repeat our 1-channel spectrogram 3 times.
X = X[..., np.newaxis]
X = np.repeat(X, 3, axis=-1) 

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.15)

# 3. LOAD THE "GIANT'S BRAIN" (MobileNetV2)
# We load a model pre-trained on millions of images, but leave off the "head" (the final decision layer)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False # We "freeze" the giant's brain so it doesn't forget what it knows

# 4. BUILD THE NEW SONGPITCH HEAD (With Resizing!)
model = models.Sequential([
    # This NEW layer shrinks your 128x1292 audio into a 128x128 square
    layers.Resizing(128, 128), 
    
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. THE PRO TRAINING
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

print("🚀 Starting Pro-Transfer Training. This model will be MUCH smarter!")
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), 
          batch_size=32, callbacks=[early_stop])

model.save('song_pitch_model.h5')
print("✨ Professional-grade model saved!")