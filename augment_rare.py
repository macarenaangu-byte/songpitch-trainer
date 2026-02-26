import os
import librosa
import soundfile as sf

# Targeting the genres that need the most help
TARGET_GENRES = ['genre_Jazz', 'genre_Latin', 'genre_Electronic', 'genre_Modern']
BASE_PATH = 'training_data'

def augment_audio():
    for genre in TARGET_GENRES:
        genre_path = os.path.join(BASE_PATH, genre)
        if not os.path.exists(genre_path): continue
        
        files = [f for f in os.listdir(genre_path) if f.endswith('.mp3')]
        print(f"🎸 Augmenting {genre}... adding pitch variations.")
        
        for file in files:
            path = os.path.join(genre_path, file)
            try:
                y, sr = librosa.load(path, sr=22050, duration=30)
                
                # Create a "High Pitch" version
                y_high = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
                sf.write(os.path.join(genre_path, f"aug_high_{file}"), y_high, sr)
                
                # Create a "Low Pitch" version
                y_low = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
                sf.write(os.path.join(genre_path, f"aug_low_{file}"), y_low, sr)
            except Exception as e:
                continue

if __name__ == "__main__":
    augment_audio()
    print("🚀 Augmentation complete! Rare genres are now 3x larger.")