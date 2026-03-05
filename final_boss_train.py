import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import LabelEncoder

# 1. DATA PREP
with open('dual_features.pkl', 'rb') as f:
    data = pickle.load(f)

X = np.repeat(np.array(data['features'])[..., np.newaxis], 3, axis=-1)
X = tf.image.resize(X, [128, 128]).numpy()

# Separate encoders for Genres and Moods
gen_enc, mood_enc = LabelEncoder(), LabelEncoder()
y_gen = gen_enc.fit_transform(data['genres'])
y_mood = mood_enc.fit_transform(data['moods'])

# 2. DUAL-HEAD ARCHITECTURE
base = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base.trainable = True

inputs = layers.Input(shape=(128, 128, 3))
x = base(inputs)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)

# Two independent prediction branches
gen_out = layers.Dense(len(gen_enc.classes_), activation='softmax', name='genre')(x)
mood_out = layers.Dense(len(mood_enc.classes_), activation='softmax', name='mood')(x)

model = models.Model(inputs=inputs, outputs=[gen_out, mood_out])

# 3. MULTI-LOSS COMPILER
model.compile(optimizer=tf.keras.optimizers.Adam(5e-5),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. SAVE MODEL AND BOTH ENCODERS
model.save('song_pitch_dual_model.h5')
with open('encoders.pkl', 'wb') as f:
    pickle.dump({'gen': gen_enc, 'mood': mood_enc}, f)

print("🚀 Specialist Training Started...")
model.fit(X, [y_gen, y_mood], epochs=60, validation_split=0.15)