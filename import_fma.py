"""
import_fma.py — Import FMA (Free Music Archive) tracks into the training_data/ structure.

USAGE:
  1. Download FMA Small (~7.2GB): https://os.unil.cloud.switch.ch/fma/fma_small.zip
  2. Download metadata: https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
  3. Extract both into this directory:
       songpitch-trainer/
         fma_small/       (contains 000/ 001/ ... 155/ subfolders with MP3s)
         fma_metadata/    (contains tracks.csv, genres.csv, etc.)
  4. Run:  python import_fma.py

This script reads the FMA metadata, maps genres to your training categories,
and copies MP3s into training_data/genre/<GenreName>/
"""

import os
import shutil
import pandas as pd

# ─── PATHS ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FMA_AUDIO_DIR = os.path.join(BASE_DIR, 'fma_small')
FMA_METADATA_DIR = os.path.join(BASE_DIR, 'fma_metadata')
TRACKS_CSV = os.path.join(FMA_METADATA_DIR, 'tracks.csv')
OUTPUT_BASE = os.path.join(BASE_DIR, 'training_data', 'genre')

# ─── FMA GENRE → YOUR GENRE MAPPING ─────────────────────────────────────────
# FMA Small has 8 root genres: Hip-Hop, Pop, Folk, Experimental, Rock,
# International, Electronic, Instrumental
# Map them to your existing training categories.

FMA_TO_YOUR_GENRE = {
    'Rock': 'Rock',
    'Pop': 'Pop',
    'Folk': 'Folk',
    'Electronic': 'Electronic',
    'Instrumental': 'Orchestral',
    'International': 'Latin',
    'Experimental': 'Alternative',
    'Hip-Hop': 'R&B',
    # Jazz, Blues, Country etc. appear as sub-genres in the full FMA
    # We'll also check the track_genres column for these
}

# Sub-genre keywords → your categories (checked against genre names in metadata)
SUBGENRE_KEYWORDS = {
    'Jazz': 'Jazz',
    'Blues': 'Jazz',       # Close enough for training
    'Soul': 'R&B',
    'R&B': 'R&B',
    'Funk': 'R&B',
    'Latin': 'Latin',
    'Reggae': 'Latin',
    'Salsa': 'Latin',
    'Country': 'Folk',
    'Singer-Songwriter': 'Folk',
    'Ambient': 'Electronic',
    'Classical': 'Orchestral',
    'Soundtrack': 'Film Score',
}

# How many samples to import per category (0 = unlimited)
# Focus on underrepresented classes
TARGET_COUNTS = {
    'R&B': 100,
    'Latin': 100,
    'Alternative': 80,
    'Jazz': 80,
    'Folk': 80,
    'Rock': 50,          # Already have 110
    'Pop': 50,           # Already have 289
    'Electronic': 0,     # Already have 443, skip
    'Orchestral': 0,     # Already have 392, skip
    'Film Score': 50,
}


def get_track_path(track_id):
    """Convert a track ID (e.g., 2) to its file path (e.g., fma_small/000/000002.mp3)"""
    tid = str(track_id).zfill(6)
    subfolder = tid[:3]
    return os.path.join(FMA_AUDIO_DIR, subfolder, f'{tid}.mp3')


def load_fma_tracks():
    """Load FMA tracks.csv with genre info."""
    print("📂 Loading FMA metadata...")

    if not os.path.exists(TRACKS_CSV):
        print(f"❌ tracks.csv not found at: {TRACKS_CSV}")
        print("   Download from: https://os.unil.cloud.switch.ch/fma/fma_metadata.zip")
        return None

    # FMA tracks.csv has 2 header rows — row 0 is category, row 1 is field name
    tracks = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1])

    # Extract the columns we need
    result = pd.DataFrame({
        'genre_top': tracks[('track', 'genre_top')],
        'genres_all': tracks[('track', 'genres_all')] if ('track', 'genres_all') in tracks.columns else '',
    })

    print(f"   Loaded {len(result)} tracks")
    return result


def count_existing():
    """Count how many files already exist per genre in training_data/"""
    counts = {}
    if os.path.exists(OUTPUT_BASE):
        for genre_folder in os.listdir(OUTPUT_BASE):
            genre_path = os.path.join(OUTPUT_BASE, genre_folder)
            if os.path.isdir(genre_path):
                counts[genre_folder] = len([f for f in os.listdir(genre_path) if f.endswith('.mp3')])
    return counts


def import_fma():
    tracks = load_fma_tracks()
    if tracks is None:
        return

    if not os.path.exists(FMA_AUDIO_DIR):
        print(f"❌ FMA audio not found at: {FMA_AUDIO_DIR}")
        print("   Download from: https://os.unil.cloud.switch.ch/fma/fma_small.zip")
        return

    existing = count_existing()
    print(f"\n📊 Current training data counts:")
    for genre, count in sorted(existing.items()):
        print(f"   {genre}: {count}")

    imported = {genre: 0 for genre in TARGET_COUNTS}

    # Pass 1: Map by top-level genre
    print("\n🚀 Importing FMA tracks...")
    for track_id, row in tracks.iterrows():
        genre_top = str(row.get('genre_top', '')).strip()

        # Determine your target genre
        target_genre = FMA_TO_YOUR_GENRE.get(genre_top)

        # Also check sub-genre keywords in genres_all
        if not target_genre:
            genres_str = str(row.get('genres_all', ''))
            for keyword, mapped_genre in SUBGENRE_KEYWORDS.items():
                if keyword.lower() in genres_str.lower():
                    target_genre = mapped_genre
                    break

        if not target_genre:
            continue

        # Check if we've hit the target for this genre
        target = TARGET_COUNTS.get(target_genre, 0)
        if target == 0:
            continue
        if imported[target_genre] >= target:
            continue

        # Check if the audio file exists
        src_path = get_track_path(track_id)
        if not os.path.exists(src_path):
            continue

        # Copy to training_data/genre/<GenreName>/
        dest_dir = os.path.join(OUTPUT_BASE, target_genre)
        os.makedirs(dest_dir, exist_ok=True)

        dest_filename = f"fma_{str(track_id).zfill(6)}.mp3"
        dest_path = os.path.join(dest_dir, dest_filename)

        if os.path.exists(dest_path):
            continue

        try:
            shutil.copy2(src_path, dest_path)
            imported[target_genre] += 1
        except Exception as e:
            print(f"   ⚠️ Error copying track {track_id}: {e}")

    # Summary
    print("\n✅ Import complete!")
    print("   Files imported per genre:")
    for genre, count in sorted(imported.items()):
        if count > 0:
            prev = existing.get(genre, 0)
            print(f"   {genre}: +{count} (was {prev}, now {prev + count})")

    # Show final counts
    final = count_existing()
    print("\n📊 Final training data counts:")
    for genre, count in sorted(final.items()):
        print(f"   {genre}: {count}")


if __name__ == '__main__':
    import_fma()
