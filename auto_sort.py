import os
import shutil
import pandas as pd
import csv

# --- PATHS ---
AUDIO_DIR = os.path.abspath('./mtg-jamendo-dataset/data/audios')
MOOD_TSV = os.path.abspath('./mtg-jamendo-dataset/data/autotagging_moodtheme.tsv')
GENRE_TSV = os.path.abspath('./mtg-jamendo-dataset/data/autotagging_genre.tsv')
OUTPUT_BASE = os.path.abspath('./training_data')

def sort_data(tsv_path, mapping, category_label):
    print(f"\n🚀 Processing {category_label}...")
    if not os.path.exists(tsv_path):
        print(f"❌ TSV missing: {tsv_path}")
        return

    df = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip', quoting=csv.QUOTE_NONE)
    df.columns = [c.lower() for c in df.columns]

    for folder_name, tags in mapping.items():
        count = 0
        for tag in tags:
            # Search for the tag in the metadata
            matches = df[df.apply(lambda row: row.astype(str).str.contains(tag, case=False).any(), axis=1)]
            
            for _, row in matches.iterrows():
                # Get the ID (e.g., '1009600')
                track_id = os.path.basename(str(row['path'])).split('.')[0]
                # Determine subfolder (e.g., '00')
                subfolder = str(row['path']).split('/')[0] if '/' in str(row['path']) else "00"
                
                # The "Magic Link": Adding .low.mp3
                filename = f"{track_id}.low.mp3"
                src = os.path.join(AUDIO_DIR, subfolder, filename)

                if os.path.exists(src):
                    dest_dir = os.path.join(OUTPUT_BASE, category_label.lower(), folder_name)
                    os.makedirs(dest_dir, exist_ok=True)
                    try:
                        shutil.move(src, os.path.join(dest_dir, filename))
                        count += 1
                    except Exception:
                        pass

        print(f"✅ {folder_name}: Successfully moved {count} files")

# Full Mappings for your Project
MOOD_MAP = {
    "Happy": ["happy", "uplifting"],
    "Melancholic": ["sad", "melancholy", "emotional"],
    "Energetic": ["energetic", "action", "epic"],
    "Atmospheric": ["atmospheric", "ambient"],
    "Suspense": ["suspense", "horror", "drama"],
    "Calm": ["calm", "relaxing", "peaceful"],
    "Romantic": ["romantic", "love", "sexy"] 
}

GENRE_MAP = {
    "Orchestral": ["orchestral", "classical"],
    "Film Score": ["soundtrack", "cinematic", "film"],
    "Musical Theater": ["musical", "theater", "vocal"],
    "Pop": ["pop", "dance"],
    "Rock": ["rock", "metal"],
    "Jazz": ["jazz", "swing", "blues"],
    "Latin": ["latin", "salsa", "reggaeton"],
    "Alternative": ["alternative", "indie"],
    "R&B": ["rnb", "soul", "funk"],
    "Folk": ["folk", "acoustic", "country"],
    "Electronic": ["electronic", "techno", "house", "synth"]
}

sort_data(MOOD_TSV, MOOD_MAP, "Mood")
sort_data(GENRE_TSV, GENRE_MAP, "Genre")
print("\n🎉 ALL DONE! Check your folders.")