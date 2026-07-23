"""
ingest_labeled_audio.py — Import hand-labeled audio into the YAMNet training dataset.

USAGE:
  python ingest_labeled_audio.py <path/to/labeled_folder>

  The labeled folder should contain one subfolder per genre.
  Subfolder names are matched case-insensitively (spaces and underscores OK):

    my_labeled_audio/
      Folk/          → genre_Folk
      Bossa Nova/    → genre_Bossa_Nova
      Afrobeats/     → genre_Afrobeats
      Jazz/          → genre_Jazz
      Latin/         → genre_Latin
      ...

  Supported audio: .mp3  .wav  .ogg  .flac  .aiff  .m4a

WHAT IT DOES:
  1. Extracts 1024-dim YAMNet embeddings for each audio file
  2. Merges them into yamnet_features.pkl (backs up the old one first)
  3. Shows before/after class counts

AFTER RUNNING:
  python train_yamnet_classifier.py
  → produces new yamnet_genre_model.h5 + yamnet_genre_model_encoder.pkl

  Then commit those two files and push to deploy.
"""

import os
import sys
import shutil
import pickle
import time
from collections import Counter

import numpy as np
import librosa
import tensorflow as tf

# ── Folder name → training label mapping ────────────────────────────────────
# Keys are lowercase; spaces and underscores are interchangeable.
FOLDER_TO_LABEL = {
    'acoustic':           'genre_Acoustic',
    'afrobeats':          'genre_Afrobeats',
    'afropop':            'genre_Afrobeats',
    'alternative rock':   'genre_Alternative_Rock',
    'ambient':            'genre_Ambient',
    'bachata':            'genre_Bachata',
    'baroque':            'genre_Baroque',
    'blues':              'genre_Blues',
    'bossa nova':         'genre_Bossa_Nova',
    'bossanova':          'genre_Bossa_Nova',
    "children's":         'genre_Childrens',
    'childrens':          'genre_Childrens',
    'cinematic':          'genre_Cinematic',
    'classical':          'genre_Classical',
    'corporate':          'genre_Corporate',
    'country':            'genre_Country',
    'cumbia':             'genre_Cumbia',
    'dancehall':          'genre_Dancehall',
    'drum and bass':      'genre_Drum_and_Bass',
    'dnb':                'genre_Drum_and_Bass',
    'dubstep':            'genre_Dubstep',
    'edm':                'genre_EDM',
    'electronic':         'genre_Electronic',
    'film score':         'genre_Film_Score',
    'flamenco':           'genre_Flamenco',
    'folk':               'genre_Folk',
    'funk soul':          'genre_Funk_Soul',
    'funk/soul':          'genre_Funk_Soul',
    'gospel':             'genre_Gospel',
    'grunge':             'genre_Grunge',
    'hard rock':          'genre_Hard_Rock',
    'hip-hop':            'genre_Hip-Hop',
    'hip hop':            'genre_Hip-Hop',
    'hiphop':             'genre_Hip-Hop',
    'house':              'genre_House',
    'hyperpop':           'genre_HyperPop',
    'indie':              'genre_Indie',
    'jazz':               'genre_Jazz',
    'k-pop':              'genre_KPop',
    'kpop':               'genre_KPop',
    'latin':              'genre_Latin',
    'lo-fi':              'genre_Lo_Fi',
    'lo fi':              'genre_Lo_Fi',
    'lofi':               'genre_Lo_Fi',
    'merengue':           'genre_Merengue',
    'metal':              'genre_Metal',
    'musical theatre':    'genre_Musical_Theatre',
    'musical theater':    'genre_Musical_Theatre',
    'new age':            'genre_New_Age',
    'opera':              'genre_Opera',
    'pop':                'genre_Pop',
    'progressive rock':   'genre_Progressive_Rock',
    'prog rock':          'genre_Progressive_Rock',
    'punk':               'genre_Punk',
    'r&b':                'genre_R&B',
    'rnb':                'genre_R&B',
    'reggae':             'genre_Reggae',
    'reggaeton':          'genre_Reggaeton',
    'reggaetón':          'genre_Reggaeton',
    'rock':               'genre_Rock',
    'salsa':              'genre_Salsa',
    'samba':              'genre_Samba',
    'mambo':              'genre_Latin',
    'cha-cha':            'genre_Latin',
    'cha cha':            'genre_Latin',
    'swing':              'genre_Jazz',
    'big band':           'genre_Jazz',
    'soul':               'genre_Funk_Soul',
    'disco':              'genre_Funk_Soul',
    'ska':                'genre_Reggae',
    'synth-pop':          'genre_Electronic',
    'synth pop':          'genre_Electronic',
    'new wave':           'genre_Electronic',
    'synthwave':          'genre_Synthwave',
    'tango':              'genre_Tango',
    'techno':             'genre_Techno',
    'trance':             'genre_Trance',
    'trap':               'genre_Trap',
    'trap latino':        'genre_Trap_Latino',
    'urbano':             'genre_Urbano',
    'world music':        'genre_World_Music',
    'world':              'genre_World_Music',
    'latin folk':         'genre_Latin_Folk',
    'waltz':              'genre_Waltz',
    'ballad':             'genre_Ballad',
    'bolero':             'genre_Bolero',
}

AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.flac', '.aiff', '.m4a'}
FEATURES_FILE    = 'yamnet_features.pkl'
YAMNET_DIR       = 'yamnet_model'
SAMPLE_RATE      = 16000
DURATION         = 30   # seconds per clip (matches existing training data)


def normalize_folder_name(name):
    return name.lower().replace('_', ' ').strip()


def load_yamnet():
    print(f"Loading YAMNet from {YAMNET_DIR} ...")
    model = tf.saved_model.load(YAMNET_DIR)
    infer = model.signatures["serving_default"]
    return infer


def extract_embedding(infer, filepath):
    """Return a 1024-dim mean embedding or None on error."""
    audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True, duration=DURATION)
    # Pad to exactly DURATION seconds
    target = DURATION * SAMPLE_RATE
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]
    audio = audio.astype(np.float32)
    peak = np.abs(audio).max()
    if peak > 0:
        audio /= peak
    out = infer(waveform=tf.constant(audio))
    embeddings = out["output_1"].numpy()   # (frames, 1024)
    return embeddings.mean(axis=0)         # (1024,)


def load_existing():
    if not os.path.exists(FEATURES_FILE):
        print(f"  No existing {FEATURES_FILE} found — starting fresh.")
        return [], [], []
    with open(FEATURES_FILE, 'rb') as f:
        data = pickle.load(f)
    return list(data['features']), list(data['labels']), list(data['filenames'])


def save_features(features, labels, filenames):
    backup = FEATURES_FILE + '.bak'
    if os.path.exists(FEATURES_FILE):
        shutil.copy2(FEATURES_FILE, backup)
        print(f"  Backed up old {FEATURES_FILE} → {backup}")
    with open(FEATURES_FILE, 'wb') as f:
        pickle.dump({
            'features':  features,
            'labels':    labels,
            'filenames': filenames,
        }, f)
    print(f"  Saved {FEATURES_FILE}  ({len(features)} total samples)")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"ERROR: '{input_dir}' is not a directory.")
        sys.exit(1)

    # ── Discover folders ──────────────────────────────────────────────────────
    folders = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if not os.path.isdir(path):
            continue
        key = normalize_folder_name(name)
        label = FOLDER_TO_LABEL.get(key)
        if label is None:
            print(f"  SKIP '{name}' — unknown genre name. "
                  f"Add it to FOLDER_TO_LABEL or rename the folder.")
            continue
        audio_files = [
            f for f in sorted(os.listdir(path))
            if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
        ]
        if not audio_files:
            print(f"  SKIP '{name}' — no audio files found.")
            continue
        folders.append((name, label, path, audio_files))

    if not folders:
        print("No recognizable genre folders found. Nothing to do.")
        sys.exit(0)

    print(f"\nFound {len(folders)} genre folder(s):")
    total_files = 0
    for name, label, _, files in folders:
        print(f"  {name:30s} → {label}  ({len(files)} files)")
        total_files += len(files)
    print(f"\nTotal audio files to process: {total_files}")

    # ── Load existing data ────────────────────────────────────────────────────
    print(f"\nLoading existing training data ...")
    orig_features, orig_labels, orig_filenames = load_existing()
    before_counts = Counter(orig_labels)
    print(f"  Existing samples: {len(orig_labels)}")
    if before_counts:
        print(f"  Existing classes: {len(before_counts)}")

    # Index: filename → list of (index, label) — used to detect re-labeling
    # (same audio file previously stored under a different genre)
    fname_index = {}
    for idx, (fn, lbl) in enumerate(zip(orig_filenames, orig_labels)):
        fname_index.setdefault(fn, []).append((idx, lbl))

    # ── Load YAMNet ───────────────────────────────────────────────────────────
    print()
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    infer = load_yamnet()

    # ── Extract new embeddings ────────────────────────────────────────────────
    new_features, new_labels, new_filenames = [], [], []
    indices_to_remove = set()   # old pkl entries that are being replaced
    added = skipped = relabeled = errors = 0
    t0 = time.time()

    for folder_name, label, folder_path, audio_files in folders:
        print(f"\n── {folder_name} ({label}) ─────────────────────────────")
        for i, fname in enumerate(audio_files):
            existing = fname_index.get(fname, [])
            same_label = any(lbl == label for _, lbl in existing)
            diff_label = [(idx, lbl) for idx, lbl in existing if lbl != label]

            if same_label and not diff_label:
                print(f"  [{i+1}/{len(audio_files)}] SKIP (already in dataset): {fname}")
                skipped += 1
                continue

            filepath = os.path.join(folder_path, fname)
            try:
                emb = extract_embedding(infer, filepath)
                new_features.append(emb)
                new_labels.append(label)
                new_filenames.append(fname)

                if diff_label:
                    # File was previously stored under a different label — remove old entry
                    old_lbls = [lbl for _, lbl in diff_label]
                    for idx, _ in diff_label:
                        indices_to_remove.add(idx)
                    relabeled += 1
                    print(f"  [{i+1}/{len(audio_files)}] RELABELED  {fname}  "
                          f"({', '.join(old_lbls)} → {label})")
                else:
                    added += 1
                    print(f"  [{i+1}/{len(audio_files)}] OK  {fname}")

                # Update index so multi-label siblings don't double-remove
                fname_index[fname] = [(None, label)]

            except Exception as e:
                errors += 1
                print(f"  [{i+1}/{len(audio_files)}] ERROR  {fname}: {e}")

    elapsed = time.time() - t0

    # ── Merge: filter out replaced entries, append new ones ───────────────────
    if indices_to_remove:
        keep = [i for i in range(len(orig_features)) if i not in indices_to_remove]
        orig_features  = [orig_features[i]  for i in keep]
        orig_labels    = [orig_labels[i]    for i in keep]
        orig_filenames = [orig_filenames[i] for i in keep]
        print(f"\n  Removed {len(indices_to_remove)} stale entries from existing dataset.")

    features  = list(orig_features)  + new_features
    labels    = list(orig_labels)    + new_labels
    filenames = list(orig_filenames) + new_filenames

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  INGESTION COMPLETE  ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"  Added new:    {added}")
    print(f"  Re-labeled:   {relabeled}  (old entry replaced with corrected label)")
    print(f"  Skipped:      {skipped}  (already in dataset with same label)")
    print(f"  Errors:       {errors}")

    if added == 0 and relabeled == 0:
        print("\n  Nothing new to save.")
        sys.exit(0)

    # Save
    print()
    save_features(np.array(features), labels, filenames)

    # Before / after per-class counts
    after_counts = Counter(labels)
    print(f"\n  Class counts (before → after):")
    all_classes = sorted(set(list(before_counts.keys()) + list(after_counts.keys())))
    for cls in all_classes:
        b = before_counts.get(cls, 0)
        a = after_counts.get(cls, 0)
        diff = f"+{a-b}" if a > b else ("" if a == b else str(a-b))
        marker = " ←" if a != b else ""
        print(f"    {cls:<35} {b:>5} → {a:>5}  {diff}{marker}")

    print(f"\n  Next step:")
    print(f"    python train_yamnet_classifier.py")
    print(f"  Then commit the new model files and push to deploy.")
