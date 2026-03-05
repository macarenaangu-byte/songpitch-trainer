"""
expand_training_data.py — Expand training data from FMA Small + Jamendo datasets.

Sources training clips for all 20 genres and 14 moods:

FROM FMA Small (8,000 tracks, 8 root genres):
  - Hip-Hop, EDM, Ambient, Indie, World Music, Reggae, Afrobeats

FROM Jamendo (13,265 downloaded tracks with mood/genre tags):
  - Blues, Country (genres)
  - Dark, Epic, Playful, Aggressive, Nostalgic, Mysterious, Triumphant (moods)

OPTIONAL (requires Jamendo API client_id in .env):
  - K-Pop, Musical Theatre (very niche, not in local datasets)

USAGE:
  python expand_training_data.py
"""

import os
import ast
import shutil
import random
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ─── PATHS ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FMA_AUDIO_DIR = os.path.join(BASE_DIR, 'fma_small')
FMA_METADATA_DIR = os.path.join(BASE_DIR, 'fma_metadata')
TRACKS_CSV = os.path.join(FMA_METADATA_DIR, 'tracks.csv')
GENRES_CSV = os.path.join(FMA_METADATA_DIR, 'genres.csv')

JAMENDO_DIR = os.path.join(BASE_DIR, 'mtg-jamendo-dataset')
JAMENDO_AUDIO_DIR = os.path.join(JAMENDO_DIR, 'data', 'audios')
JAMENDO_MOOD_TSV = os.path.join(JAMENDO_DIR, 'data', 'autotagging_moodtheme.tsv')
JAMENDO_GENRE_TSV = os.path.join(JAMENDO_DIR, 'data', 'autotagging_genre.tsv')

TRAINING_DIR = os.path.join(BASE_DIR, 'training_data')

# ─── TARGET COUNTS PER CLASS ────────────────────────────────────────────────
TARGET_PER_CLASS = 150

# ─── FMA GENRE ID → SUB-GENRE MAPPING ───────────────────────────────────────
# These genre IDs come from fma_metadata/genres.csv
# We use the genres_all column (list of genre IDs) to find sub-genre matches

FMA_GENRE_MAPPINGS = {
    # genre_Hip-Hop: FMA root genre Hip-Hop (21) + sub-genres
    'genre_Hip-Hop': {
        'genre_top': ['Hip-Hop'],
        'genre_ids': [21, 100, 83, 539, 542, 580, 693, 811],  # Hip-Hop, Alt Hip-Hop, Nerdcore, Rap, Breakbeat, Abstract, Wonky, Beats
    },
    # genre_EDM: Electronic sub-genres (Techno, House, Dance, Dubstep, Trance-like)
    'genre_EDM': {
        'genre_top': [],
        'genre_ids': [181, 182, 296, 468, 401, 337, 695, 491],  # Techno, House, Dance, Dubstep, Bigbeat, Drum&Bass, Jungle, Skweee
    },
    # genre_Ambient: Ambient Electronic (42) + Ambient (107) from Instrumental
    'genre_Ambient': {
        'genre_top': [],
        'genre_ids': [42, 107, 267, 400, 495],  # Ambient Electronic, Ambient, New Age, Chill-out, Downtempo
    },
    # genre_Indie: Indie-Rock (66) + Lo-Fi (27) + Singer-Songwriter (103)
    'genre_Indie': {
        'genre_top': [],
        'genre_ids': [66, 27, 111],  # Indie-Rock, Lo-Fi, Power-Pop
    },
    # genre_World_Music: International root (2) but exclude Reggae/Afrobeat-specific
    'genre_World_Music': {
        'genre_top': ['International'],
        'genre_ids': [2, 46, 77, 86, 102, 118, 130, 172, 176, 177, 232, 502, 504, 619, 741],
        'exclude_ids': [79, 602, 81, 92, 214],  # Exclude Reggae-Dub, Dancehall, Afrobeat, African, N.African
    },
    # genre_Reggae: Reggae - Dub (79) + Reggae - Dancehall (602)
    'genre_Reggae': {
        'genre_top': [],
        'genre_ids': [79, 602],  # Reggae-Dub, Reggae-Dancehall
    },
    # genre_Afrobeats: Afrobeat (81) + African (92) + North African (214)
    'genre_Afrobeats': {
        'genre_top': [],
        'genre_ids': [81, 92, 214],  # Afrobeat, African, North African
    },
}

# ─── JAMENDO GENRE TAG MAPPINGS ─────────────────────────────────────────────
# Maps Jamendo genre tags (from autotagging_genre.tsv) to our training folders
JAMENDO_GENRE_MAPPINGS = {
    'genre_Blues': ['blues', 'bluesrock'],
    'genre_Country': ['country'],
    'genre_Afrobeats': ['african', 'ethno', 'tribal'],  # Supplement FMA data
}

# ─── JAMENDO MOOD TAG MAPPINGS ──────────────────────────────────────────────
# Maps Jamendo mood/theme tags to our training folders
JAMENDO_MOOD_MAPPINGS = {
    'mood_Dark': ['dark', 'darkambient', 'horror'],
    'mood_Epic': ['epic', 'dramatic', 'trailer'],
    'mood_Playful': ['fun', 'funny', 'children'],
    'mood_Aggressive': ['action', 'heavy', 'hard'],
    'mood_Nostalgic': ['retro', 'mellow', 'melancholic'],
    'mood_Mysterious': ['space', 'soundscape', 'horror'],
    'mood_Triumphant': ['powerful', 'uplifting', 'motivational'],
}

# ─── JAMENDO API SETTINGS (optional) ────────────────────────────────────────
JAMENDO_CLIENT_ID = os.getenv('JAMENDO_CLIENT_ID', '')
JAMENDO_API_GENRES = {
    'genre_K-Pop': ['kpop', 'korean', 'pop+asian'],
    'genre_Musical_Theatre': ['musical', 'broadway', 'showtunes'],
}


def get_fma_track_path(track_id):
    """Convert FMA track ID to file path: fma_small/000/000002.mp3"""
    tid = str(track_id).zfill(6)
    return os.path.join(FMA_AUDIO_DIR, tid[:3], f'{tid}.mp3')


def get_jamendo_audio_path(tsv_path):
    """Convert Jamendo TSV path (48/948.mp3) to actual file path."""
    return os.path.join(JAMENDO_AUDIO_DIR, tsv_path.replace('.mp3', '.low.mp3'))


def count_existing(folder):
    """Count MP3 files in a training_data folder."""
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith('.mp3')])


def copy_file(src, dest_dir, prefix, track_id):
    """Copy an audio file to the destination directory."""
    os.makedirs(dest_dir, exist_ok=True)
    dest_name = f"{prefix}_{track_id}.mp3"
    dest_path = os.path.join(dest_dir, dest_name)
    if os.path.exists(dest_path):
        return False
    try:
        shutil.copy2(src, dest_path)
        return True
    except Exception as e:
        print(f"  ⚠️  Error copying {src}: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  PART 1: Import from FMA Small
# ═══════════════════════════════════════════════════════════════════════════

def import_from_fma():
    """Import genre training data from FMA Small dataset."""
    print("\n" + "=" * 60)
    print("  PART 1: Importing from FMA Small")
    print("=" * 60)

    if not os.path.exists(TRACKS_CSV):
        print("❌ FMA tracks.csv not found. Skipping FMA import.")
        return
    if not os.path.exists(FMA_AUDIO_DIR):
        print("❌ FMA audio dir not found. Skipping FMA import.")
        return

    # Load FMA metadata
    print("\n📂 Loading FMA metadata...")
    tracks = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1])

    # Get track IDs that exist in fma_small
    fma_ids = set()
    for sub in os.listdir(FMA_AUDIO_DIR):
        subpath = os.path.join(FMA_AUDIO_DIR, sub)
        if os.path.isdir(subpath):
            for f in os.listdir(subpath):
                if f.endswith('.mp3'):
                    fma_ids.add(int(f.replace('.mp3', '')))
    print(f"   Found {len(fma_ids)} tracks in fma_small/")

    # Load genre ID → title map
    genres_df = pd.read_csv(GENRES_CSV)
    gid_to_title = dict(zip(genres_df['genre_id'], genres_df['title']))

    # Process each target genre
    for folder_name, mapping in FMA_GENRE_MAPPINGS.items():
        dest_dir = os.path.join(TRAINING_DIR, folder_name)
        existing = count_existing(dest_dir)
        needed = max(0, TARGET_PER_CLASS - existing)

        if needed == 0:
            print(f"\n✅ {folder_name}: Already has {existing} tracks (target: {TARGET_PER_CLASS})")
            continue

        print(f"\n🔍 {folder_name}: Have {existing}, need {needed} more...")

        genre_top_filter = set(mapping.get('genre_top', []))
        genre_id_filter = set(mapping.get('genre_ids', []))
        exclude_ids = set(mapping.get('exclude_ids', []))

        # Find matching tracks
        candidates = []
        for track_id in fma_ids:
            if track_id not in tracks.index:
                continue

            row = tracks.loc[track_id]
            top_genre = str(row.get(('track', 'genre_top'), ''))

            # Parse genres_all (list of genre IDs)
            try:
                genres_all = ast.literal_eval(str(row.get(('track', 'genres_all'), '[]')))
            except (ValueError, SyntaxError):
                genres_all = []

            # Check exclusion first
            if exclude_ids and any(gid in exclude_ids for gid in genres_all):
                continue

            # Match by genre_top or by any genre ID in genres_all
            matched = False
            if genre_top_filter and top_genre in genre_top_filter:
                matched = True
            if not matched and genre_id_filter:
                for gid in genres_all:
                    if gid in genre_id_filter:
                        matched = True
                        break

            if matched:
                src_path = get_fma_track_path(track_id)
                if os.path.exists(src_path):
                    candidates.append((track_id, src_path))

        # Shuffle and take what we need
        random.shuffle(candidates)
        imported = 0
        for track_id, src_path in candidates[:needed]:
            if copy_file(src_path, dest_dir, 'fma', str(track_id).zfill(6)):
                imported += 1

        final = count_existing(dest_dir)
        print(f"   ✅ Imported {imported} tracks → total: {final}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 2: Import from Jamendo (local downloaded audio)
# ═══════════════════════════════════════════════════════════════════════════

def parse_jamendo_tsv(tsv_path, tag_prefix):
    """
    Parse a Jamendo autotagging TSV file.
    Returns dict: tag_name → list of (track_id, audio_path) tuples
    """
    tag_tracks = {}
    with open(tsv_path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue
            track_id = parts[0]  # e.g., track_0000948
            path = parts[3]      # e.g., 48/948.mp3
            tags = [p.replace(f'{tag_prefix}---', '') for p in parts[5:]]

            audio_path = get_jamendo_audio_path(path)
            if not os.path.exists(audio_path):
                continue

            for tag in tags:
                if tag not in tag_tracks:
                    tag_tracks[tag] = []
                tag_tracks[tag].append((track_id, audio_path))

    return tag_tracks


def import_jamendo_genres():
    """Import genre training data from Jamendo dataset."""
    print("\n" + "=" * 60)
    print("  PART 2a: Importing Genres from Jamendo")
    print("=" * 60)

    if not os.path.exists(JAMENDO_GENRE_TSV):
        print("❌ Jamendo genre TSV not found. Skipping.")
        return

    print("\n📂 Parsing Jamendo genre tags...")
    tag_tracks = parse_jamendo_tsv(JAMENDO_GENRE_TSV, 'genre')

    for folder_name, tag_list in JAMENDO_GENRE_MAPPINGS.items():
        dest_dir = os.path.join(TRAINING_DIR, folder_name)
        existing = count_existing(dest_dir)
        needed = max(0, TARGET_PER_CLASS - existing)

        if needed == 0:
            print(f"\n✅ {folder_name}: Already has {existing} tracks (target: {TARGET_PER_CLASS})")
            continue

        print(f"\n🔍 {folder_name}: Have {existing}, need {needed} more...")

        # Collect all candidates from matching tags
        candidates = []
        seen_ids = set()
        for tag in tag_list:
            for track_id, audio_path in tag_tracks.get(tag, []):
                if track_id not in seen_ids:
                    seen_ids.add(track_id)
                    candidates.append((track_id, audio_path))

        random.shuffle(candidates)
        imported = 0
        for track_id, audio_path in candidates[:needed]:
            clean_id = track_id.replace('track_', '')
            if copy_file(audio_path, dest_dir, 'jam', clean_id):
                imported += 1

        final = count_existing(dest_dir)
        print(f"   ✅ Imported {imported} tracks → total: {final}")


def import_jamendo_moods():
    """Import mood training data from Jamendo dataset."""
    print("\n" + "=" * 60)
    print("  PART 2b: Importing Moods from Jamendo")
    print("=" * 60)

    if not os.path.exists(JAMENDO_MOOD_TSV):
        print("❌ Jamendo mood TSV not found. Skipping.")
        return

    print("\n📂 Parsing Jamendo mood tags...")
    tag_tracks = parse_jamendo_tsv(JAMENDO_MOOD_TSV, 'mood/theme')

    for folder_name, tag_list in JAMENDO_MOOD_MAPPINGS.items():
        dest_dir = os.path.join(TRAINING_DIR, folder_name)
        existing = count_existing(dest_dir)
        needed = max(0, TARGET_PER_CLASS - existing)

        if needed == 0:
            print(f"\n✅ {folder_name}: Already has {existing} tracks (target: {TARGET_PER_CLASS})")
            continue

        print(f"\n🔍 {folder_name}: Have {existing}, need {needed} more...")

        # Collect all candidates from matching tags
        candidates = []
        seen_ids = set()
        for tag in tag_list:
            for track_id, audio_path in tag_tracks.get(tag, []):
                if track_id not in seen_ids:
                    seen_ids.add(track_id)
                    candidates.append((track_id, audio_path))

        random.shuffle(candidates)
        imported = 0
        for track_id, audio_path in candidates[:needed]:
            clean_id = track_id.replace('track_', '')
            if copy_file(audio_path, dest_dir, 'jam', clean_id):
                imported += 1

        final = count_existing(dest_dir)
        print(f"   ✅ Imported {imported} tracks → total: {final}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 3: Download from Jamendo API (for K-Pop, Musical Theatre)
# ═══════════════════════════════════════════════════════════════════════════

def import_jamendo_api():
    """Download tracks from Jamendo API for niche genres not in local datasets."""
    print("\n" + "=" * 60)
    print("  PART 3: Jamendo API Downloads (K-Pop, Musical Theatre)")
    print("=" * 60)

    if not JAMENDO_CLIENT_ID:
        print("\n⚠️  No JAMENDO_CLIENT_ID in .env — skipping API downloads.")
        print("   To enable: Register at https://devportal.jamendo.com/")
        print("   Then add JAMENDO_CLIENT_ID=your_id to .env")
        print("   K-Pop and Musical Theatre will use rule-based classification as fallback.")
        return

    try:
        import requests
    except ImportError:
        print("❌ 'requests' library not installed. Run: pip install requests")
        return

    for folder_name, search_tags in JAMENDO_API_GENRES.items():
        dest_dir = os.path.join(TRAINING_DIR, folder_name)
        existing = count_existing(dest_dir)
        needed = max(0, TARGET_PER_CLASS - existing)

        if needed == 0:
            print(f"\n✅ {folder_name}: Already has {existing} tracks")
            continue

        print(f"\n🔍 {folder_name}: Have {existing}, need {needed} more...")

        downloaded = 0
        for tag in search_tags:
            if downloaded >= needed:
                break

            url = "https://api.jamendo.com/v3.0/tracks/"
            params = {
                'client_id': JAMENDO_CLIENT_ID,
                'format': 'json',
                'limit': min(100, needed - downloaded),
                'tags': tag,
                'include': 'musicinfo',
                'audioformat': 'mp3',
            }

            try:
                resp = requests.get(url, params=params, timeout=30)
                data = resp.json()

                for track in data.get('results', []):
                    if downloaded >= needed:
                        break

                    audio_url = track.get('audiodownload') or track.get('audio')
                    if not audio_url:
                        continue

                    track_id = str(track['id'])
                    dest_path = os.path.join(dest_dir, f"jamapi_{track_id}.mp3")
                    if os.path.exists(dest_path):
                        continue

                    os.makedirs(dest_dir, exist_ok=True)
                    try:
                        audio_resp = requests.get(audio_url, timeout=60)
                        with open(dest_path, 'wb') as f:
                            f.write(audio_resp.content)
                        downloaded += 1
                        print(f"   ⬇️  Downloaded {track['name'][:40]}...")
                    except Exception as e:
                        print(f"   ⚠️  Download error: {e}")

            except Exception as e:
                print(f"   ⚠️  API error for tag '{tag}': {e}")

        final = count_existing(dest_dir)
        print(f"   ✅ Downloaded {downloaded} tracks → total: {final}")


# ═══════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def print_summary():
    """Print final summary of all training data."""
    print("\n" + "=" * 60)
    print("  FINAL TRAINING DATA SUMMARY")
    print("=" * 60)

    genre_total = 0
    mood_total = 0

    print("\n📊 GENRES:")
    for folder in sorted(os.listdir(TRAINING_DIR)):
        if folder.startswith('genre_'):
            count = count_existing(os.path.join(TRAINING_DIR, folder))
            status = "✅" if count >= 80 else "⚠️ LOW" if count >= 30 else "❌ INSUFFICIENT"
            clean_name = folder.replace('genre_', '').replace('_', ' ')
            print(f"   {status} {clean_name:20s} {count:4d} tracks")
            genre_total += count

    print(f"\n   Total genre tracks: {genre_total}")

    print("\n📊 MOODS:")
    for folder in sorted(os.listdir(TRAINING_DIR)):
        if folder.startswith('mood_'):
            count = count_existing(os.path.join(TRAINING_DIR, folder))
            status = "✅" if count >= 80 else "⚠️ LOW" if count >= 30 else "❌ INSUFFICIENT"
            clean_name = folder.replace('mood_', '').replace('_', ' ')
            print(f"   {status} {clean_name:20s} {count:4d} tracks")
            mood_total += count

    print(f"\n   Total mood tracks: {mood_total}")
    print(f"\n   Grand total: {genre_total + mood_total} tracks")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("🎵 SongPitch Training Data Expansion")
    print("   Target: 20 genres + 14 moods")
    print(f"   Target per class: {TARGET_PER_CLASS}")

    import_from_fma()
    import_jamendo_genres()
    import_jamendo_moods()
    import_jamendo_api()
    print_summary()

    print("\n🎉 Done! Next steps:")
    print("   1. python extract_features.py    (extract mel-spectrograms)")
    print("   2. python improved_train.py       (retrain models)")
