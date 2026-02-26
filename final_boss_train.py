import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

# 1. LOAD DATA
print("📂 Loading the refined library...")
with open('audio_features.pkl', 'rb') as f:
    data = pickle.load(f)

X = np.array(data['features'])[..., np.newaxis]
X = np.repeat(X, 3, axis=-1)
X = tf.image.resize(X, [128, 128]).numpy()

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(data['labels'])

# 2. SPLIT & WEIGHTS
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.15, stratify=y_encoded, random_state=42
)

weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
weight_dict = dict(enumerate(weights))

# 3. BUILD THE BRAIN (Clean Version)
base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = True 

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4), # Slightly lower dropout to help accuracy climb faster
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(len(encoder.classes_), activation='softmax')
])

# 4. BALANCED COMPILER
# Going back to a proven speed that won't get stuck at 7%
model.compile(optimizer=tf.keras.optimizers.Adam(5e-5), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. HIGH-PATIENCE CALLBACKS
checkpoint = callbacks.ModelCheckpoint('song_pitch_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
# Increased patience to 15 so it doesn't stop too early
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# 6. TRAIN
print(f"🚀 Final Accuracy Push on {len(encoder.classes_)} classes...")
model.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test),
          class_weight=weight_dict, callbacks=[checkpoint, early_stop])

print("✨ Training Complete. Check the morning report for the new scores!")