"""
extract_yamnet_features.py — Extract YAMNet embeddings from training audio.

Replaces mel-spectrogram extraction with YAMNet (pre-trained on AudioSet,
2M+ audio clips, 521 classes). YAMNet outputs 1024-dim embeddings per
0.96-second segment. We mean-pool across segments to get a single 1024-dim
vector per track.

USAGE:
  pip install tensorflow-hub
  python extract_yamnet_features.py

OUTPUT:
  yamnet_features.pkl — {features: (N, 1024), labels: [...], filenames: [...]}
"""

import os
import re
import pickle
import time
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub

# ─── CONFIG ──────────────────────────────────────────────────────────────────
TRAINING_DIR = 'training_data'
OUTPUT_FILE = 'yamnet_features.pkl'
CHECKPOINT_FILE = 'yamnet_extraction_checkpoint.pkl'
SAMPLE_RATE = 16000  # YAMNet expects 16kHz mono audio
DURATION = 30        # seconds per clip
CHECKPOINT_EVERY = 200  # save progress every N tracks

# ─── LOAD YAMNET ─────────────────────────────────────────────────────────────
print("🔄 Loading YAMNet model from TF Hub...")
t0 = time.time()
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
print(f"✅ YAMNet loaded in {time.time() - t0:.1f}s")

# ─── RESUME SUPPORT ──────────────────────────────────────────────────────────
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, 'rb') as f:
        checkpoint = pickle.load(f)
    features = checkpoint['features']
    labels = checkpoint['labels']
    filenames = checkpoint['filenames']
    processed_set = set(checkpoint.get('processed_keys', []))
    print(f"📂 Resuming from checkpoint: {len(features)} tracks already processed")
else:
    features = []
    labels = []
    filenames = []
    processed_set = set()
    print("📂 Starting fresh extraction")

# ─── COLLECT ALL FILES ───────────────────────────────────────────────────────
all_files = []
for category in sorted(os.listdir(TRAINING_DIR)):
    cat_dir = os.path.join(TRAINING_DIR, category)
    if not os.path.isdir(cat_dir) or not (category.startswith('genre_') or category.startswith('mood_')):
        continue
    for fname in sorted(os.listdir(cat_dir)):
        if fname.lower().endswith(('.mp3', '.wav', '.ogg', '.flac')):
            all_files.append((category, fname, os.path.join(cat_dir, fname)))

total = len(all_files)
to_process = [(cat, fn, fp) for cat, fn, fp in all_files if f"{cat}/{fn}" not in processed_set]
print(f"📊 Total files: {total}, Already processed: {total - len(to_process)}, Remaining: {len(to_process)}")

# ─── EXTRACT EMBEDDINGS ─────────────────────────────────────────────────────
target_length = DURATION * SAMPLE_RATE  # 480,000 samples
errors = 0
t_start = time.time()

for i, (category, fname, filepath) in enumerate(to_process):
    try:
        # Load audio at 16kHz mono
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)

        # Pad or trim to exactly 30 seconds
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]

        # Ensure float32 in [-1, 1]
        y = y.astype(np.float32)
        if np.abs(y).max() > 0:
            y = y / max(np.abs(y).max(), 1.0)

        # Run YAMNet — returns (scores, embeddings, spectrogram)
        scores, embeddings, spectrogram = yamnet_model(y)

        # Mean pool across time segments → single 1024-dim vector
        embedding = np.mean(embeddings.numpy(), axis=0)  # (1024,)

        features.append(embedding)
        labels.append(category)
        filenames.append(fname)
        processed_set.add(f"{category}/{fname}")

    except Exception as e:
        errors += 1
        print(f"  ⚠️ Error processing {filepath}: {e}")
        continue

    # Progress reporting
    if (i + 1) % 50 == 0 or (i + 1) == len(to_process):
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed
        remaining = (len(to_process) - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1}/{len(to_process)}] {category}/{fname} — "
              f"{rate:.1f} tracks/sec, ~{remaining/60:.0f} min remaining")

    # Checkpoint
    if (i + 1) % CHECKPOINT_EVERY == 0:
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump({
                'features': features,
                'labels': labels,
                'filenames': filenames,
                'processed_keys': list(processed_set),
            }, f)
        print(f"  💾 Checkpoint saved ({len(features)} tracks)")

# ─── SAVE FINAL OUTPUT ───────────────────────────────────────────────────────
features_array = np.array(features)
print(f"\n{'='*60}")
print(f"  EXTRACTION COMPLETE")
print(f"{'='*60}")
print(f"  Total tracks: {len(features)}")
print(f"  Errors: {errors}")
print(f"  Feature shape: {features_array.shape}")
print(f"  Time: {(time.time() - t_start) / 60:.1f} minutes")

# Show per-class counts
from collections import Counter
print(f"\n📊 Per-class distribution:")
for label, count in sorted(Counter(labels).items()):
    print(f"   {label}: {count}")

# Save
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump({
        'features': features_array,
        'labels': labels,
        'filenames': filenames,
    }, f)
print(f"\n💾 Saved {OUTPUT_FILE} ({features_array.nbytes / 1024 / 1024:.1f} MB)")

# Clean up checkpoint
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("🧹 Checkpoint file cleaned up")

print("\n🎉 Done! Run train_yamnet_classifier.py next.")
