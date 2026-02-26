import os
import librosa
import numpy as np
import pickle

# --- SETTINGS ---
DATA_PATH = '/Users/macarena.nadeau/Desktop/songpitch-trainer/training_data'
OUTPUT_FILE = 'audio_features.pkl'
SAMPLE_RATE = 22050
DURATION = 30 

def extract_features():
    features = []
    labels = []
    filenames = []

    print("🚀 Starting Deep-Dive Feature Extraction...")

    # We look at every folder inside training_data
    for category_folder in os.listdir(DATA_PATH):
        category_path = os.path.join(DATA_PATH, category_folder)
        
        if os.path.isdir(category_path):
            print(f"📂 Scanning Category: {category_folder}")
            
            # This "walks" through every subfolder inside (like genre_Jazz/Jazz/)
            for root, dirs, files in os.walk(category_path):
                for file in files:
                    if file.endswith('.mp3'):
                        file_path = os.path.join(root, file)
                        
                        try:
                            y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

                            target_length = DURATION * SAMPLE_RATE
                            if len(y) < target_length:
                                y = np.pad(y, (0, target_length - len(y)))

                            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
                            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                            
                            features.append(mel_spec_db)
                            # We use the top-level folder name as the label
                            labels.append(category_folder)
                            filenames.append(file)
                            
                            if len(features) % 10 == 0:
                                print(f"✅ Total Found: {len(features)}", end="\r")
                                
                        except Exception as e:
                            continue

    print(f"\n💾 Saving {len(features)} features to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump({'features': features, 'labels': labels, 'filenames': filenames}, f)
    
    print("✨ EXTRACTION COMPLETE!")

if __name__ == "__main__":
    extract_features()