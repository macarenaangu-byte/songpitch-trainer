import os
import librosa
import numpy as np
import pickle

# --- SETTINGS ---
DATA_PATH = 'training_data'  # Uses relative path for GitHub
OUTPUT_FILE = 'audio_features.pkl'
SAMPLE_RATE = 22050
DURATION = 30


def extract_dual_features():
    features, labels, filenames = [], [], []

    print("🚀 Extracting Mel-Spectrogram Features...")
    print(f"   Source: {os.path.abspath(DATA_PATH)}")
    print(f"   Output: {OUTPUT_FILE}")

    for category in sorted(os.listdir(DATA_PATH)):
        cat_path = os.path.join(DATA_PATH, category)
        if not os.path.isdir(cat_path):
            continue

        # Only process genre_ and mood_ folders
        if not (category.startswith('genre_') or category.startswith('mood_')):
            continue

        cat_count = 0
        for root, _, files in os.walk(cat_path):
            for file in files:
                if file.endswith('.mp3'):
                    try:
                        y, sr = librosa.load(os.path.join(root, file), sr=SAMPLE_RATE, duration=DURATION)
                        # Ensure uniform length
                        y = np.pad(y, (0, max(0, DURATION * SAMPLE_RATE - len(y))))[:DURATION * SAMPLE_RATE]

                        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr))
                        features.append(mel)

                        # Single label per sample (the folder name, e.g., genre_Rock or mood_Happy)
                        labels.append(category)
                        filenames.append(file)
                        cat_count += 1

                        if len(features) % 10 == 0:
                            print(f"✅ Total Found: {len(features)}", end="\r")
                    except Exception:
                        continue

        print(f"   📁 {category}: {cat_count} tracks")

    # Save in format compatible with improved_train.py
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump({
            'features': features,
            'labels': labels,
            'filenames': filenames,
        }, f)

    # Summary
    from collections import Counter
    label_counts = Counter(labels)
    genre_count = sum(v for k, v in label_counts.items() if k.startswith('genre_'))
    mood_count = sum(v for k, v in label_counts.items() if k.startswith('mood_'))
    print(f"\n✨ EXTRACTION COMPLETE!")
    print(f"   {len(features)} total samples")
    print(f"   {genre_count} genre samples across {sum(1 for k in label_counts if k.startswith('genre_'))} classes")
    print(f"   {mood_count} mood samples across {sum(1 for k in label_counts if k.startswith('mood_'))} classes")
    print(f"   Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_dual_features()
