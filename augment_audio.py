"""
augment_audio.py — Audio-level data augmentation for underrepresented classes.

Creates augmented copies of audio files using pitch shifting, time stretching,
and noise injection. Unlike spectrogram-level augmentation, these create
genuinely different audio that YAMNet will encode differently.

TARGET: Bring every class to at least 400 samples (from current 85-150 for
underrepresented classes). Also augment ALL classes to improve generalization.

NAMING: aug_pitch_high_, aug_pitch_low_, aug_stretch_slow_, aug_stretch_fast_,
aug_noise_ — get_base_song() strips these for group-aware splitting.

USAGE:
  python augment_audio.py
"""

import os
import numpy as np
import librosa
import soundfile as sf
from collections import Counter
import time

TRAINING_DIR = 'training_data'
SAMPLE_RATE = 22050  # librosa default
DURATION = 30

# Classes below this threshold get FULL augmentation (5 variants each)
HEAVY_AUG_THRESHOLD = 200
# All classes get at least LIGHT augmentation (2 variants each)
LIGHT_AUG_THRESHOLD = 500


def augment_file(filepath, output_dir, prefix, augment_fn):
    """Load audio, apply augmentation, save as WAV."""
    try:
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)
        if len(y) < sr * 2:  # skip very short files
            return False

        y_aug = augment_fn(y, sr)

        # Ensure same length
        target_len = DURATION * sr
        if len(y_aug) < target_len:
            y_aug = np.pad(y_aug, (0, target_len - len(y_aug)))
        else:
            y_aug = y_aug[:target_len]

        # Normalize
        if np.abs(y_aug).max() > 0:
            y_aug = y_aug / np.abs(y_aug).max() * 0.95

        basename = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(output_dir, f"{prefix}{basename}.wav")
        sf.write(out_path, y_aug, sr)
        return True
    except Exception as e:
        print(f"    ⚠️ Error augmenting {filepath}: {e}")
        return False


def pitch_shift_high(y, sr):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=2)

def pitch_shift_low(y, sr):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)

def time_stretch_slow(y, sr):
    return librosa.effects.time_stretch(y, rate=0.9)

def time_stretch_fast(y, sr):
    return librosa.effects.time_stretch(y, rate=1.1)

def add_noise(y, sr):
    noise = np.random.normal(0, 0.005, len(y))
    return y + noise.astype(np.float32)

def pitch_shift_minor_high(y, sr):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=1)

def pitch_shift_minor_low(y, sr):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=-1)


# ─── MAIN ────────────────────────────────────────────────────────────────────
print("📊 Scanning existing training data...\n")

# Count existing files per class
class_counts = {}
class_files = {}
for category in sorted(os.listdir(TRAINING_DIR)):
    cat_dir = os.path.join(TRAINING_DIR, category)
    if not os.path.isdir(cat_dir) or not (category.startswith('genre_') or category.startswith('mood_')):
        continue

    # Only count ORIGINAL files (not already augmented)
    originals = []
    for f in os.listdir(cat_dir):
        if f.lower().endswith(('.mp3', '.wav', '.ogg', '.flac')):
            if not f.startswith('aug_'):
                originals.append(os.path.join(cat_dir, f))

    class_counts[category] = len(originals)
    class_files[category] = originals

# Show current distribution
print("Current distribution (originals only):")
for cat, count in sorted(class_counts.items(), key=lambda x: x[1]):
    level = "🔴" if count < HEAVY_AUG_THRESHOLD else "🟡" if count < LIGHT_AUG_THRESHOLD else "🟢"
    print(f"  {level} {cat}: {count}")

# Define augmentation strategies per class
heavy_augmentations = [
    ('aug_pitch_high_', pitch_shift_high),
    ('aug_pitch_low_', pitch_shift_low),
    ('aug_stretch_slow_', time_stretch_slow),
    ('aug_stretch_fast_', time_stretch_fast),
    ('aug_noise_', add_noise),
    ('aug_pitch_minor_high_', pitch_shift_minor_high),
    ('aug_pitch_minor_low_', pitch_shift_minor_low),
]

light_augmentations = [
    ('aug_pitch_high_', pitch_shift_high),
    ('aug_pitch_low_', pitch_shift_low),
]

print(f"\n{'='*60}")
print(f"  STARTING AUGMENTATION")
print(f"{'='*60}")
print(f"  Heavy augmentation (7 variants): classes < {HEAVY_AUG_THRESHOLD} samples")
print(f"  Light augmentation (2 variants): classes < {LIGHT_AUG_THRESHOLD} samples")
print(f"  No augmentation: classes >= {LIGHT_AUG_THRESHOLD} samples")

t_start = time.time()
total_created = 0

for category in sorted(class_counts.keys()):
    count = class_counts[category]
    files = class_files[category]
    cat_dir = os.path.join(TRAINING_DIR, category)

    if count < HEAVY_AUG_THRESHOLD:
        augmentations = heavy_augmentations
        level = "HEAVY"
    elif count < LIGHT_AUG_THRESHOLD:
        augmentations = light_augmentations
        level = "LIGHT"
    else:
        print(f"\n⏭️  {category}: {count} samples — skipping (>= {LIGHT_AUG_THRESHOLD})")
        continue

    expected = count * len(augmentations)
    print(f"\n🔧 {category}: {count} originals × {len(augmentations)} augmentations "
          f"= {expected} new files ({level})")

    created = 0
    for i, filepath in enumerate(files):
        for prefix, aug_fn in augmentations:
            # Check if already exists
            basename = os.path.splitext(os.path.basename(filepath))[0]
            out_path = os.path.join(cat_dir, f"{prefix}{basename}.wav")
            if os.path.exists(out_path):
                created += 1
                continue

            if augment_file(filepath, cat_dir, prefix, aug_fn):
                created += 1

        if (i + 1) % 25 == 0:
            print(f"    [{i+1}/{count}] {created} augmented files created...")

    total_created += created
    print(f"  ✅ {category}: {created} augmented files created (total now: {count + created})")

elapsed = time.time() - t_start
print(f"\n{'='*60}")
print(f"  AUGMENTATION COMPLETE")
print(f"{'='*60}")
print(f"  Total augmented files created: {total_created}")
print(f"  Time: {elapsed/60:.1f} minutes")

# Show new distribution
print(f"\n📊 Updated distribution (originals + augmented):")
for category in sorted(class_counts.keys()):
    cat_dir = os.path.join(TRAINING_DIR, category)
    total = len([f for f in os.listdir(cat_dir) if f.lower().endswith(('.mp3', '.wav', '.ogg', '.flac'))])
    orig = class_counts[category]
    print(f"  {category}: {orig} originals + {total - orig} augmented = {total} total")

print(f"\n🎉 Done! Re-run extract_yamnet_features.py next.")
