"""
save_norm_params.py — Extract and save the exact normalization parameters
that were used during training, so inference preprocessing matches exactly.
"""
import pickle
import numpy as np
import tensorflow as tf

print("📂 Loading audio_features.pkl...")
with open('audio_features.pkl', 'rb') as f:
    data = pickle.load(f)

X_raw = data['features']
print(f"   {len(X_raw)} samples loaded")

# Replicate EXACT same preprocessing as improved_train.py
X = np.array(X_raw)[..., np.newaxis]          # (N, 128, W, 1)
X = np.repeat(X, 3, axis=-1)                   # (N, 128, W, 3)
X = tf.image.resize(X, [128, 128]).numpy()      # (N, 128, 128, 3)

X_min = float(X.min())
X_max = float(X.max())

print(f"\n📏 Normalization params:")
print(f"   X_min = {X_min}")
print(f"   X_max = {X_max}")
print(f"   (hardcoded in main.py was: -80.0, 0.0)")

# Save
with open('norm_params.pkl', 'wb') as f:
    pickle.dump({'X_min': X_min, 'X_max': X_max}, f)

print(f"\n💾 Saved to norm_params.pkl")
