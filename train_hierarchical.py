"""
train_hierarchical.py — Two-stage hierarchical genre classifier on Discogs features.

Loads 2641-dim Discogs-EfficientNet feature vectors from extract_discogs_features.py,
applies StandardScaler normalisation (critical — features span very different scales),
then trains:
  Stage 1: Classify into 10 broad categories
  Stage 2: Classify into specific sub-genre within the predicted category

With Discogs-EffNet features the combined end-to-end accuracy target is 75-90%.

USAGE:
  python train_hierarchical.py

INPUTS:
  discogs_features.pkl — from extract_discogs_features.py

OUTPUTS:
  feature_scaler.pkl          — StandardScaler (must ship with models)
  stage1_model.h5 + stage1_model_encoder.pkl
  stage2_<Category>_model.h5 + stage2_<Category>_model_encoder.pkl  (x10)
"""

import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from collections import Counter


# ── Hierarchy definition ──────────────────────────────────────────────────────
HIERARCHY = {
    'Latin':               ['genre_Bachata', 'genre_Cumbia', 'genre_Merengue', 'genre_Tango',
                            'genre_Flamenco', 'genre_Reggaeton', 'genre_Dancehall', 'genre_Latin',
                            'genre_Trap_Latino', 'genre_Urbano', 'genre_Salsa', 'genre_Samba',
                            'genre_Bossa_Nova'],
    'Electronic':          ['genre_Techno', 'genre_EDM', 'genre_Synthwave',
                            'genre_House', 'genre_Trance', 'genre_Dubstep',
                            'genre_Drum_and_Bass', 'genre_Electronic'],
    'Rock_Metal':          ['genre_Rock', 'genre_Metal', 'genre_Punk', 'genre_Grunge',
                            'genre_Hard_Rock', 'genre_Alternative_Rock', 'genre_Progressive_Rock'],
    'Classical_Cinematic': ['genre_Classical', 'genre_Opera', 'genre_Baroque',
                            'genre_Film_Score', 'genre_Cinematic'],
    'HipHop_Urban':        ['genre_Hip-Hop', 'genre_Trap', 'genre_R&B', 'genre_HyperPop', 'genre_Afrobeats'],
    'Pop_Indie':           ['genre_Pop', 'genre_Indie', 'genre_KPop', 'genre_Acoustic'],
    'Folk_Country_Roots':  ['genre_Folk', 'genre_Country', 'genre_Reggae', 'genre_World_Music'],
    'Jazz_Blues':          ['genre_Jazz', 'genre_Blues', 'genre_Funk_Soul', 'genre_Gospel'],
    'Ambient_Chill':       ['genre_Ambient', 'genre_Lo_Fi', 'genre_New_Age', 'genre_Corporate'],
    'Theatrical':          ['genre_Musical_Theatre', 'genre_Childrens'],
}

# Reverse lookup: specific genre → broad category
GENRE_TO_CATEGORY = {
    genre: cat
    for cat, genres in HIERARCHY.items()
    for genre in genres
}


def get_base_song(filename):
    """Strip augmentation prefixes to recover the original song filename."""
    return re.sub(
        r'^(aug_pitch_high_|aug_pitch_low_|aug_stretch_slow_|aug_stretch_fast_|'
        r'aug_noise_|aug_quiet_|aug_loud_|aug_high_|aug_low_)+',
        '', filename
    )


def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for class-imbalanced datasets.

    Down-weights easy examples so training focuses on hard / rare ones.
    """
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(y_true_int, depth=num_classes)
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = alpha * y_true_one_hot * tf.pow(1 - y_pred, gamma)
        focal = tf.reduce_sum(weight * cross_entropy, axis=-1)
        return focal
    return loss_fn


def cosine_decay_schedule(initial_lr=1e-3, min_lr=1e-6, total_steps=1000):
    """Cosine annealing LR: starts at initial_lr, smoothly decays to min_lr."""
    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps,
        alpha=min_lr / initial_lr,
    )


def build_and_train(X, y_raw, groups, model_name, num_epochs=150,
                    min_samples_per_class=5, wide_model=False,
                    use_focal_loss=True):
    """Train a Dense classifier on 1024-dim YAMNet embeddings.

    Args:
        X:          (N, 1024) feature matrix
        y_raw:      (N,) string labels
        groups:     (N,) base-song IDs for group-aware splitting
        model_name: stem for saved files  (model_name.h5, model_name_encoder.pkl)
        num_epochs: maximum training epochs (early stopping usually kicks in sooner)
        min_samples_per_class: classes with fewer samples are dropped
        wide_model: if True, use a wider 1024→512→256 architecture (for Stage 1)
        use_focal_loss: if False, use label-smoothed cross-entropy (better for
                        well-balanced multi-class problems like Stage 1)

    Returns:
        (model, encoder, best_val_accuracy)  — model/encoder are None if skipped
    """
    print(f"\n{'='*60}")
    print(f"  TRAINING: {model_name}")
    print(f"{'='*60}")

    # ── Show class distribution ──
    label_counts = Counter(y_raw)
    print(f"\n📊 Class distribution ({len(label_counts)} classes):")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"   {label}: {count}")

    # ── Drop micro-classes that can't be split meaningfully ──
    valid_labels = {lbl for lbl, cnt in label_counts.items()
                    if cnt >= min_samples_per_class}
    if len(valid_labels) < len(label_counts):
        dropped = set(label_counts.keys()) - valid_labels
        print(f"\n⚠️  Dropping {len(dropped)} class(es) with < {min_samples_per_class} "
              f"samples: {dropped}")
        mask = np.array([y in valid_labels for y in y_raw])
        X, y_raw, groups = X[mask], y_raw[mask], groups[mask]

    if len(set(y_raw)) < 2:
        print(f"⚠️  Only 1 class remains — skipping {model_name}")
        return None, None, 0.0

    # ── Encode labels ──
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)
    print(f"\n🏷️  {num_classes} classes: {list(encoder.classes_)}")

    encoder_file = f'{model_name}_encoder.pkl'
    with open(encoder_file, 'wb') as f:
        pickle.dump(encoder, f)
    print(f"💾 Saved {encoder_file}")

    # ── Group-aware train / test split ──
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_encoded, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    train_unique = len(set(groups[train_idx]))
    test_unique  = len(set(groups[test_idx]))
    overlap      = set(groups[train_idx]) & set(groups[test_idx])
    print(f"\n📐 Split: {len(X_train)} train ({train_unique} unique songs) / "
          f"{len(X_test)} test ({test_unique} unique songs)")

    # ── Feature-space augmentation (replaces audio augmentation files) ────────
    # Add Gaussian noise (σ=0.02 on normalised features) to create synthetic
    # copies — equivalent to the pitch/stretch/noise audio augmentations, but
    # done in milliseconds instead of hours of re-extraction.
    N_AUG = 4   # 4 noisy copies + 1 original = 5x training data
    rng   = np.random.default_rng(42)
    aug_X = [X_train]
    aug_y = [y_train]
    for _ in range(N_AUG):
        noise = rng.normal(0, 0.02, X_train.shape).astype(np.float32)
        aug_X.append(X_train + noise)
        aug_y.append(y_train)
    X_train = np.concatenate(aug_X, axis=0)
    y_train = np.concatenate(aug_y, axis=0)
    shuffle_idx = rng.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    print(f"   After feature augmentation: {len(X_train):,} training samples ({N_AUG+1}x)")
    if overlap:
        print(f"⚠️  WARNING: {len(overlap)} songs appear in both train and test!")
    else:
        print(f"✅ No data leakage — 0 songs shared between train and test")

    # Test-set class distribution
    print(f"\n📊 Test set distribution:")
    for cls_idx in range(num_classes):
        cls_mask   = y_test == cls_idx
        cls_groups = set(groups[test_idx][cls_mask])
        print(f"   {encoder.classes_[cls_idx]}: {cls_mask.sum()} samples "
              f"({len(cls_groups)} unique songs)")

    # ── Class weights (capped at 8x) ──
    weights    = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_encoded), y=y_encoded
    )
    weight_dict = dict(enumerate(weights))
    min_w       = min(weight_dict.values())
    weight_dict = {k: min(v, min_w * 8) for k, v in weight_dict.items()}

    print(f"\n⚖️  Class weights:")
    for idx, w in sorted(weight_dict.items()):
        print(f"   {encoder.classes_[idx]}: {w:.2f}")

    # ── Build model ──
    # Input shape is inferred from X so the same function works for any feature dim.
    # wide_model=True (Stage 1): deeper, more capacity for broad 10-category task
    # wide_model=False (Stage 2): lighter, fine-grained within a single category
    feat_dim = X.shape[1]
    if wide_model:
        model = models.Sequential([
            layers.Input(shape=(feat_dim,)),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.35),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            layers.Dense(num_classes, activation='softmax'),
        ])
    else:
        model = models.Sequential([
            layers.Input(shape=(feat_dim,)),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax'),
        ])

    steps_per_epoch = max(1, len(X_train) // 64)
    total_steps     = num_epochs * steps_per_epoch
    lr_schedule     = cosine_decay_schedule(
        initial_lr=1e-3, min_lr=1e-6, total_steps=total_steps
    )

    # Stage 1 (10 balanced categories): standard cross-entropy works better
    #   — focal loss over-emphasises hard boundary cases and hurts the easy ones
    # Stage 2 (imbalanced sub-genres within a category): keep focal loss
    if use_focal_loss:
        loss_fn  = focal_loss(gamma=2.0, alpha=0.25)
        loss_name = 'focal loss'
    else:
        loss_fn  = tf.keras.losses.SparseCategoricalCrossentropy()
        loss_name = 'sparse cross-entropy'

    print(f"\n🔧 Loss: {loss_name} | Model: {'wide (1024→512→256)' if wide_model else 'standard (512→256→128)'}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss_fn,
        metrics=['accuracy'],
    )
    model.summary()

    # ── Callbacks ──
    model_file = f'{model_name}.h5'
    cb = [
        callbacks.EarlyStopping(
            monitor='val_accuracy', patience=25,
            restore_best_weights=True, verbose=1, mode='max',
        ),
        callbacks.ModelCheckpoint(
            model_file, monitor='val_accuracy',
            save_best_only=True, verbose=1, mode='max',
        ),
    ]

    # ── Train ──
    print(f"\n🚀 Training {model_name} for up to {num_epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=num_epochs,
        batch_size=64,
        class_weight=weight_dict,
        callbacks=cb,
        verbose=1,
    )

    best_acc = max(history.history.get('val_accuracy', [0]))
    print(f"\n✨ {model_name} — best val accuracy: {best_acc:.4f} ({best_acc:.1%})")

    # ── Per-class breakdown ──
    custom_objs = {'loss_fn': focal_loss(gamma=2.0, alpha=0.25)} if use_focal_loss else {}
    best_model  = tf.keras.models.load_model(model_file, custom_objects=custom_objs)
    predictions  = best_model.predict(X_test)
    pred_classes = np.argmax(predictions, axis=1)

    good_classes  = 0
    total_classes = 0
    for cls_idx in range(num_classes):
        mask = y_test == cls_idx
        if mask.sum() == 0:
            continue
        total_classes += 1
        cls_acc  = (pred_classes[mask] == cls_idx).mean()
        n_unique = len(set(groups[test_idx][mask]))
        status   = "✅" if cls_acc >= 0.5 else ("⚠️" if cls_acc >= 0.3 else "❌")
        if cls_acc >= 0.5:
            good_classes += 1
        print(f"   {status} {encoder.classes_[cls_idx]}: {cls_acc:.1%} "
              f"({mask.sum()} samples, {n_unique} unique songs)")

    print(f"\n🏆 {good_classes}/{total_classes} classes at ≥50% accuracy")

    # Classification report (only classes present in test set)
    test_classes_present = sorted(set(y_test.tolist()))
    report = classification_report(
        y_test, pred_classes,
        target_names=[encoder.classes_[i] for i in test_classes_present],
        labels=test_classes_present,
        zero_division=0,
    )
    print(f"\n📋 Classification Report:\n{report}")

    return best_model, encoder, best_acc


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("📂 Loading Discogs features...")
    with open('discogs_features.pkl', 'rb') as f:
        data = pickle.load(f)

    X_all = np.array(data['features'], dtype=np.float32)
    y_all = np.array(data['labels'])
    f_all = np.array(data['filenames'])

    print(f"\n📊 Full dataset: {len(y_all):,} samples, {len(set(y_all))} classes")
    print(f"   Feature dimensions: {X_all.shape[1]}")
    for label, count in sorted(Counter(y_all).items(), key=lambda x: -x[1]):
        print(f"   {label}: {count}")

    # ── Remove NaN / Inf rows (a handful of corrupt/silent audio files) ────────
    finite_mask = np.isfinite(X_all).all(axis=1)
    bad = (~finite_mask).sum()
    if bad > 0:
        print(f"⚠️  Dropping {bad} samples with NaN/Inf values (corrupt audio)")
        X_all = X_all[finite_mask]
        y_all = y_all[finite_mask]
        f_all = f_all[finite_mask]
    print(f"✅ Clean dataset: {len(y_all):,} samples")

    # ── Normalise features ────────────────────────────────────────────────────
    # MFCC values span ±50, spectral contrast spans 0-80 dB, RMS spans 0-0.1 —
    # wildly different scales. StandardScaler (zero mean, unit variance) levels
    # the playing field so no feature group dominates gradient updates.
    print("\n⚖️  Fitting StandardScaler on full dataset...")
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    SCALER_FILE = 'feature_scaler.pkl'
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 Saved {SCALER_FILE}  (must be deployed alongside the models)")

    # Base song IDs for group-aware splitting
    base_songs = np.array([get_base_song(fn) for fn in f_all])
    print(f"\n🔍 Unique base songs: {len(set(base_songs))}")

    # ── Genre-only subset ──────────────────────────────────────────────────────
    genre_mask = np.array([y.startswith('genre_') for y in y_all])
    X_genre = X_all[genre_mask]
    y_genre = y_all[genre_mask]
    f_genre = base_songs[genre_mask]

    print(f"\n🎸 Genre samples: {len(y_genre)} | Classes: {len(set(y_genre))}")

    # ── Warn about unmapped genres ─────────────────────────────────────────────
    all_known     = set(GENRE_TO_CATEGORY.keys())
    genres_in_data = set(y_genre)
    unmapped       = genres_in_data - all_known
    if unmapped:
        print(f"\n⚠️  Unmapped genres (will be skipped): {unmapped}")
        keep_mask = np.array([y in all_known for y in y_genre])
        X_genre   = X_genre[keep_mask]
        y_genre   = y_genre[keep_mask]
        f_genre   = f_genre[keep_mask]
        print(f"   Remaining after filter: {len(y_genre)} samples")

    # ── STAGE 1 — Broad category classifier (10 categories) ───────────────────
    print(f"\n{'='*60}")
    print(f"  STAGE 1: Broad Category Classifier  (10 categories)")
    print(f"{'='*60}")

    y_cat = np.array([GENRE_TO_CATEGORY[y] for y in y_genre])

    # Stage 1 gets a wider model + standard cross-entropy
    # (categories are balanced; focal loss hurts by over-weighting boundary songs)
    stage1_model, stage1_encoder, stage1_acc = build_and_train(
        X_genre, y_cat, f_genre,
        model_name='stage1_model',
        num_epochs=150,
        wide_model=True,
        use_focal_loss=False,
    )

    # ── STAGE 2 — Per-category sub-genre classifiers ──────────────────────────
    print(f"\n{'='*60}")
    print(f"  STAGE 2: Per-Category Sub-Genre Classifiers")
    print(f"{'='*60}")

    stage2_results = {}

    for category, genre_list in HIERARCHY.items():
        cat_mask  = np.array([y in genre_list for y in y_genre])
        n_samples = cat_mask.sum()

        if n_samples < 20:
            print(f"\n⚠️  {category}: only {n_samples} samples — skipping")
            stage2_results[category] = (None, None, 0.0)
            continue

        genres_present = set(y_genre[cat_mask])
        if len(genres_present) < 2:
            # Only one genre exists in this category: trivially 100% accurate
            print(f"\n⚠️  {category}: only 1 genre present {genres_present} — "
                  f"skipping (trivially correct)")
            # Still save encoder so main.py can decode
            enc = LabelEncoder()
            enc.fit(y_genre[cat_mask])
            enc_file = f'stage2_{category}_model_encoder.pkl'
            with open(enc_file, 'wb') as fh:
                pickle.dump(enc, fh)
            stage2_results[category] = (None, enc, 1.0)
            continue

        print(f"\n🎯 {category}: {n_samples} samples, "
              f"{len(genres_present)}/{len(genre_list)} genres present in data")

        # Stage 2 keeps focal loss — sub-genres within each category ARE imbalanced
        model, encoder, acc = build_and_train(
            X_genre[cat_mask],
            y_genre[cat_mask],
            f_genre[cat_mask],
            model_name=f'stage2_{category}_model',
            num_epochs=150,
            wide_model=False,
            use_focal_loss=True,
        )
        stage2_results[category] = (model, encoder, acc)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  HIERARCHICAL TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Stage 1 (category): {stage1_acc:.1%} val accuracy")
    print(f"\n  Stage 2 (sub-genre):")
    for cat, (mdl, enc, acc) in stage2_results.items():
        genres_present = set(y_genre[np.array([y in HIERARCHY[cat] for y in y_genre])])
        n_present      = len(genres_present)
        if mdl is None and acc == 0.0:
            print(f"    {cat}: ⚠️  skipped (< 20 samples)")
        elif mdl is None:
            print(f"    {cat}: ✅ single genre — trivially correct")
        else:
            print(f"    {cat}: {acc:.1%} val accuracy  ({n_present} genres)")

    print(f"\n  Files saved:")
    print(f"    feature_scaler.pkl              ← normalisation scaler (required for inference)")
    print(f"    stage1_model.h5 + stage1_model_encoder.pkl")
    for cat in HIERARCHY:
        _, mdl_obj, _ = stage2_results.get(cat, (None, None, 0))
        if mdl_obj is not None or stage2_results.get(cat, (None,))[0] is not None:
            print(f"    stage2_{cat}_model.h5 + stage2_{cat}_model_encoder.pkl")

    print(f"\n🎉 Hierarchical models ready!")
    print(f"\n  Next steps:")
    print(f"    1. Verify accuracy numbers above")
    print(f"    2. git add -f feature_scaler.pkl")
    print(f"       git add -f stage1_model.h5 stage1_model_encoder.pkl")
    print(f"       git add -f stage2_*_model.h5 stage2_*_model_encoder.pkl")
    print(f"       git add -f yamnet_mood_model.h5 yamnet_mood_model_encoder.pkl")
    print(f"       git add main.py")
    print(f"    3. git commit + git push → Render redeploys (~2 min)")
