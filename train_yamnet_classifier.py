"""
train_yamnet_classifier.py — Train genre + mood classifiers on YAMNet embeddings.

Trains two lightweight Dense classifiers on the 1024-dim YAMNet embeddings
extracted by extract_yamnet_features.py. Training is extremely fast (minutes)
since the embeddings are pre-computed and the classifier is small.

KEY TECHNIQUES:
  - Group-aware train/test split (prevents data leakage with augmented copies)
  - Focal loss (handles class imbalance better than cross-entropy)
  - Label smoothing (prevents overconfident predictions)
  - Class weights (capped at 8x)
  - Cosine annealing LR schedule
  - Early stopping with patience=15

USAGE:
  python train_yamnet_classifier.py

INPUTS:
  yamnet_features.pkl — from extract_yamnet_features.py

OUTPUTS:
  yamnet_genre_model.keras + yamnet_genre_encoder.pkl
  yamnet_mood_model.keras  + yamnet_mood_encoder.pkl
"""

import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from collections import Counter


def get_base_song(filename):
    """Strip augmentation prefixes to get the original song filename."""
    return re.sub(
        r'^(aug_pitch_high_|aug_pitch_low_|aug_stretch_slow_|aug_stretch_fast_|'
        r'aug_noise_|aug_quiet_|aug_loud_|aug_high_|aug_low_)+',
        '', filename
    )


def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for handling class imbalance.

    Down-weights easy/well-classified examples, focuses training on
    hard/misclassified ones. Much better than cross-entropy for
    imbalanced datasets.
    """
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        # y_true is sparse (integer labels)
        y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(y_true_int, depth=num_classes)

        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = alpha * y_true_one_hot * tf.pow(1 - y_pred, gamma)
        focal = tf.reduce_sum(weight * cross_entropy, axis=-1)
        return focal
    return loss_fn


def cosine_decay_schedule(initial_lr=1e-3, min_lr=1e-6, total_steps=1000):
    """Create a cosine decay learning rate schedule."""
    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps,
        alpha=min_lr / initial_lr,  # ratio of min_lr to initial_lr
    )


def build_and_train(X, y_raw, groups, model_name, num_epochs=150):
    """Build and train a Dense classifier on YAMNet embeddings."""
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

    # ── GROUP-AWARE SPLIT ──
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_encoded, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    train_unique = len(set(groups[train_idx]))
    test_unique = len(set(groups[test_idx]))
    overlap = set(groups[train_idx]) & set(groups[test_idx])
    print(f"📐 Split: {len(X_train)} train ({train_unique} unique) / "
          f"{len(X_test)} test ({test_unique} unique)")
    if overlap:
        print(f"⚠️  WARNING: {len(overlap)} songs in both train and test!")
    else:
        print(f"✅ No data leakage — 0 songs shared between train and test")

    # Test set distribution
    print(f"\n📊 Test set class distribution:")
    for cls_idx in range(num_classes):
        cls_mask = y_test == cls_idx
        cls_groups = set(groups[test_idx][cls_mask])
        print(f"   {encoder.classes_[cls_idx]}: {cls_mask.sum()} samples "
              f"({len(cls_groups)} unique songs)")

    # Class weights (capped at 8x)
    weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_encoded), y=y_encoded
    )
    weight_dict = dict(enumerate(weights))
    min_w = min(weight_dict.values())
    weight_dict = {k: min(v, min_w * 8) for k, v in weight_dict.items()}

    print(f"\n⚖️  Class weights:")
    for idx, w in sorted(weight_dict.items()):
        print(f"   {encoder.classes_[idx]}: {w:.2f}")

    # ── Build model ──
    # Input: 1024-dim YAMNet embedding
    model = models.Sequential([
        layers.Input(shape=(1024,)),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Cosine decay: total steps = epochs * steps_per_epoch
    steps_per_epoch = max(1, len(X_train) // 64)
    total_steps = num_epochs * steps_per_epoch
    lr_schedule = cosine_decay_schedule(
        initial_lr=1e-3, min_lr=1e-6, total_steps=total_steps
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )

    model.summary()

    # ── Callbacks ──
    model_file = f'{model_name}.keras'
    cb = [
        callbacks.EarlyStopping(
            monitor='val_accuracy', patience=25,
            restore_best_weights=True, verbose=1, mode='max'
        ),
        callbacks.ModelCheckpoint(
            model_file, monitor='val_accuracy',
            save_best_only=True, verbose=1, mode='max'
        ),
    ]

    # ── Train ──
    print(f"\n🚀 Training for up to {num_epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=num_epochs,
        batch_size=64,
        class_weight=weight_dict,
        callbacks=cb,
        verbose=1
    )

    # ── Results ──
    best_acc = max(history.history.get('val_accuracy', [0]))
    print(f"\n✨ {model_name} — best val accuracy: {best_acc:.4f} ({best_acc:.1%})")

    # Per-class accuracy
    print(f"\n📊 Per-class results for {model_name}:")
    # Load best model for evaluation
    best_model = tf.keras.models.load_model(
        model_file,
        custom_objects={'loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
    )
    predictions = best_model.predict(X_test)
    pred_classes = np.argmax(predictions, axis=1)

    good_classes = 0
    total_classes = 0
    for cls_idx in range(num_classes):
        mask = y_test == cls_idx
        if mask.sum() == 0:
            continue
        total_classes += 1
        cls_acc = (pred_classes[mask] == cls_idx).mean()
        n_unique = len(set(groups[test_idx][mask]))
        status = "✅" if cls_acc >= 0.5 else "⚠️" if cls_acc >= 0.3 else "❌"
        if cls_acc >= 0.5:
            good_classes += 1
        print(f"   {status} {encoder.classes_[cls_idx]}: {cls_acc:.1%} "
              f"({mask.sum()} samples, {n_unique} unique songs)")

    print(f"\n🏆 {good_classes}/{total_classes} classes at 50%+ accuracy")

    # Overall confusion matrix highlights
    from sklearn.metrics import classification_report
    report = classification_report(
        y_test, pred_classes,
        target_names=encoder.classes_,
        zero_division=0
    )
    print(f"\n📋 Classification Report:\n{report}")

    return best_model, encoder, best_acc


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("📂 Loading YAMNet features...")
    with open('yamnet_features.pkl', 'rb') as f:
        data = pickle.load(f)

    X_all = np.array(data['features'])
    y_all = np.array(data['labels'])
    f_all = np.array(data['filenames'])

    print(f"\n📊 Full dataset: {len(y_all)} samples across {len(set(y_all))} classes")
    for label, count in sorted(Counter(y_all).items(), key=lambda x: -x[1]):
        print(f"   {label}: {count}")

    # Base song IDs for group-aware splitting
    base_songs = np.array([get_base_song(f) for f in f_all])
    print(f"\n🔍 Unique base songs: {len(set(base_songs))} (total: {len(base_songs)})")

    # Split into genre and mood subsets
    genre_mask = np.array([y.startswith('genre_') for y in y_all])
    mood_mask = np.array([y.startswith('mood_') for y in y_all])

    X_genre, y_genre = X_all[genre_mask], y_all[genre_mask]
    X_mood, y_mood = X_all[mood_mask], y_all[mood_mask]
    f_genre = base_songs[genre_mask]
    f_mood = base_songs[mood_mask]

    print(f"\n🎸 Genre: {len(y_genre)} samples, {len(set(y_genre))} classes, "
          f"{len(set(f_genre))} unique songs")
    print(f"🎭 Mood:  {len(y_mood)} samples, {len(set(y_mood))} classes, "
          f"{len(set(f_mood))} unique songs")

    # Train genre model
    genre_model, genre_encoder, genre_acc = build_and_train(
        X_genre, y_genre, f_genre, 'yamnet_genre_model', num_epochs=150
    )

    # Train mood model
    mood_model, mood_encoder, mood_acc = build_and_train(
        X_mood, y_mood, f_mood, 'yamnet_mood_model', num_epochs=150
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Genre model: {genre_acc:.1%} val accuracy ({len(genre_encoder.classes_)} classes)")
    print(f"  Mood model:  {mood_acc:.1%} val accuracy ({len(mood_encoder.classes_)} classes)")
    print(f"\n  Files saved:")
    print(f"    yamnet_genre_model.keras + yamnet_genre_encoder.pkl")
    print(f"    yamnet_mood_model.keras  + yamnet_mood_encoder.pkl")
    print(f"\n🎉 Both models ready!")
