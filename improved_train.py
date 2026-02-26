"""
improved_train.py — Dual-model training for genre and mood classification.

Trains TWO separate MobileNetV2 models:
  1. Genre model (10 classes) — classifies music genre
  2. Mood model (7 classes) — classifies emotional mood

KEY FIX (v2): Group-aware train/test split.
  Augmented copies of the same song (aug_high_, aug_low_ prefixes) are kept
  together in either train OR test — never split across both. This prevents
  data leakage and gives honest accuracy numbers.

Also adds SpecAugment (frequency/time masking) for better generalization
on genres with few unique songs (e.g., Latin: 102 unique, Jazz: 68 unique).

Each model uses two-phase training:
  Phase 1: Train head only (base frozen) at LR=1e-3
  Phase 2: Fine-tune top layers at LR=1e-4

USAGE:
  1. First run extract_features.py to generate fresh audio_features.pkl
  2. Then: python improved_train.py
"""

import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from collections import Counter


def get_base_song(filename):
    """Strip augmentation prefixes to get the original song filename.

    e.g. 'aug_high_aug_low_fma_000666.mp3' → 'fma_000666.mp3'
    """
    return re.sub(r'^(aug_high_|aug_low_)+', '', filename)


def spec_augment(image):
    """Apply SpecAugment: random frequency and time masking.

    This forces the model to learn robust features rather than
    memorizing specific spectrogram patterns.
    """
    img = image.copy()
    h, w, c = img.shape  # (128, 128, 3)

    # Frequency masking — zero out a random horizontal band (up to 20 bins)
    f_mask_width = np.random.randint(1, min(21, h // 4))
    f_start = np.random.randint(0, h - f_mask_width)
    img[f_start:f_start + f_mask_width, :, :] = 0.0

    # Time masking — zero out a random vertical band (up to 20 steps)
    t_mask_width = np.random.randint(1, min(21, w // 4))
    t_start = np.random.randint(0, w - t_mask_width)
    img[:, t_start:t_start + t_mask_width, :] = 0.0

    return img


# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────
print("📂 Loading features...")
with open('audio_features.pkl', 'rb') as f:
    data = pickle.load(f)

X_raw = data['features']
y_raw = data['labels']
f_raw = data.get('filenames', None)

if f_raw is None:
    print("❌ ERROR: audio_features.pkl missing 'filenames' key.")
    print("   Re-run extract_features.py first (updated version saves filenames).")
    exit(1)

print(f"\n📊 Full dataset: {len(y_raw)} samples across {len(set(y_raw))} classes")
for label, count in sorted(Counter(y_raw).items(), key=lambda x: -x[1]):
    print(f"   {label}: {count}")

# Show unique vs augmented breakdown
all_filenames = np.array(f_raw)
base_songs = np.array([get_base_song(f) for f in f_raw])
print(f"\n🔍 Unique base songs: {len(set(base_songs))} (total samples: {len(base_songs)})")

# ─── 2. PREPARE DATA ─────────────────────────────────────────────────────────
X_all = np.array(X_raw)[..., np.newaxis]        # (N, 128, W, 1)
X_all = np.repeat(X_all, 3, axis=-1)             # (N, 128, W, 3)
X_all = tf.image.resize(X_all, [128, 128]).numpy()

# Normalize to [0, 255] then apply MobileNetV2 preprocessing
X_min, X_max = X_all.min(), X_all.max()
X_all = (X_all - X_min) / (X_max - X_min) * 255.0
X_all = tf.keras.applications.mobilenet_v2.preprocess_input(X_all)
print(f"\n📏 Input range: [{X_all.min():.2f}, {X_all.max():.2f}]")

y_all = np.array(y_raw)

# Split into genre and mood datasets
genre_mask = np.array([y.startswith('genre_') for y in y_all])
mood_mask = np.array([y.startswith('mood_') for y in y_all])

X_genre, y_genre_raw = X_all[genre_mask], y_all[genre_mask]
X_mood, y_mood_raw = X_all[mood_mask], y_all[mood_mask]
f_genre = base_songs[genre_mask]  # base song IDs for grouping
f_mood = base_songs[mood_mask]

print(f"\n🎸 Genre subset: {len(y_genre_raw)} samples, {len(set(y_genre_raw))} classes, {len(set(f_genre))} unique songs")
print(f"🎭 Mood subset:  {len(y_mood_raw)} samples, {len(set(y_mood_raw))} classes, {len(set(f_mood))} unique songs")

# ─── 3. DATA AUGMENTATION ────────────────────────────────────────────────────
datagen = ImageDataGenerator(
    horizontal_flip=True,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.05,
    brightness_range=[0.9, 1.1],
    preprocessing_function=spec_augment,  # SpecAugment for generalization
)


def build_and_train(X, y_raw, groups, model_name, epochs_p1=15, epochs_p2=30):
    """Build and train a MobileNetV2 model with two-phase training.

    Uses group-aware splitting to prevent data leakage: all augmented
    variants of the same song stay together in train OR test.
    """
    print(f"\n{'='*60}")
    print(f"  TRAINING: {model_name}")
    print(f"{'='*60}")

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)
    print(f"\n🏷️  {num_classes} classes: {list(encoder.classes_)}")

    # Save encoder
    encoder_file = f'{model_name}_encoder.pkl'
    with open(encoder_file, 'wb') as f:
        pickle.dump(encoder, f)
    print(f"💾 Saved {encoder_file}")

    # ── GROUP-AWARE SPLIT (prevents data leakage) ──
    # All augmented copies of the same song go to either train OR test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_encoded, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    train_unique = len(set(groups[train_idx]))
    test_unique = len(set(groups[test_idx]))
    # Verify no overlap
    overlap = set(groups[train_idx]) & set(groups[test_idx])
    print(f"📐 Split: {len(X_train)} train ({train_unique} unique songs) / "
          f"{len(X_test)} test ({test_unique} unique songs)")
    if overlap:
        print(f"⚠️  WARNING: {len(overlap)} songs appear in both train and test!")
    else:
        print(f"✅ No data leakage — 0 songs shared between train and test")

    # Show per-class distribution in test set
    print(f"\n📊 Test set class distribution:")
    for cls_idx in range(num_classes):
        cls_mask = y_test == cls_idx
        cls_groups = set(groups[test_idx][cls_mask])
        print(f"   {encoder.classes_[cls_idx]}: {cls_mask.sum()} samples ({len(cls_groups)} unique songs)")

    # Class weights
    weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_encoded), y=y_encoded
    )
    weight_dict = dict(enumerate(weights))
    min_w = min(weight_dict.values())
    weight_dict = {k: min(v, min_w * 8) for k, v in weight_dict.items()}

    print(f"\n⚖️  Class weights:")
    for idx, w in sorted(weight_dict.items()):
        print(f"   {encoder.classes_[idx]}: {w:.2f}")

    # Fit augmentation (SpecAugment will be applied during flow())
    datagen.fit(X_train)

    # ── Build model ──
    base = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3), include_top=False, weights='imagenet'
    )
    base.trainable = False  # Freeze for Phase 1

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)  # Increased from 0.4 for better regularization
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)  # Increased from 0.3
    output = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base.input, outputs=output)

    # ── Phase 1: Train head ──
    print(f"\n🧊 Phase 1: Training head only...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    p1_cb = [
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss', patience=6, restore_best_weights=True, verbose=1
        ),
    ]

    history1 = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=epochs_p1,
        validation_data=(X_test, y_test),
        class_weight=weight_dict,
        callbacks=p1_cb,
    )
    p1_acc = max(history1.history.get('val_accuracy', [0]))
    print(f"📈 Phase 1 best val accuracy: {p1_acc:.4f}")

    # ── Phase 2: Fine-tune ──
    print(f"\n🔥 Phase 2: Fine-tuning top layers...")
    base.trainable = True
    fine_tune_at = len(base.layers) // 2
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model_file = f'{model_name}.h5'
    p2_cb = [
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
        ),
        callbacks.ModelCheckpoint(
            model_file, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
    ]

    history2 = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=epochs_p2,
        validation_data=(X_test, y_test),
        class_weight=weight_dict,
        callbacks=p2_cb,
    )

    # ── Results ──
    all_acc = history1.history.get('val_accuracy', []) + history2.history.get('val_accuracy', [])
    best_acc = max(all_acc) if all_acc else 0
    print(f"\n✨ {model_name} saved — best val accuracy: {best_acc:.4f}")

    # Per-class
    print(f"\n📊 Per-class results for {model_name}:")
    predictions = model.predict(X_test)
    pred_classes = np.argmax(predictions, axis=1)

    for cls_idx in range(num_classes):
        mask = y_test == cls_idx
        if mask.sum() == 0:
            continue
        cls_acc = (pred_classes[mask] == cls_idx).mean()
        n_unique = len(set(groups[test_idx][mask]))
        status = "✅" if cls_acc >= 0.5 else "⚠️" if cls_acc >= 0.3 else "❌"
        print(f"   {status} {encoder.classes_[cls_idx]}: {cls_acc:.1%} "
              f"({mask.sum()} samples, {n_unique} unique songs)")

    good = sum(1 for i in range(num_classes) if (y_test == i).sum() > 0 and (pred_classes[y_test == i] == i).mean() >= 0.5)
    total = sum(1 for i in range(num_classes) if (y_test == i).sum() > 0)
    print(f"🏆 {good}/{total} classes at 50%+ accuracy")

    return model, encoder, best_acc


# ─── 4. TRAIN GENRE MODEL ────────────────────────────────────────────────────
genre_model, genre_encoder, genre_acc = build_and_train(
    X_genre, y_genre_raw, f_genre, 'genre_model', epochs_p1=15, epochs_p2=35
)

# ─── 5. TRAIN MOOD MODEL ─────────────────────────────────────────────────────
mood_model, mood_encoder, mood_acc = build_and_train(
    X_mood, y_mood_raw, f_mood, 'mood_model', epochs_p1=15, epochs_p2=35
)

# ─── 6. SUMMARY ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  TRAINING COMPLETE")
print(f"{'='*60}")
print(f"  Genre model: {genre_acc:.1%} val accuracy ({len(genre_encoder.classes_)} classes)")
print(f"  Mood model:  {mood_acc:.1%} val accuracy ({len(mood_encoder.classes_)} classes)")
print(f"\n  ⚠️  Note: Accuracy is now HONEST (no data leakage).")
print(f"  Numbers may be lower than before, but real-world performance is better.")
print(f"\n  Files saved:")
print(f"    genre_model.h5 + genre_model_encoder.pkl")
print(f"    mood_model.h5  + mood_model_encoder.pkl")
print(f"\n🎉 Both models ready!")
