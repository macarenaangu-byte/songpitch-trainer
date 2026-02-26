"""
fix_structure.py — Fix training_data folder structure for proper labeling.

PROBLEM: All mood files are nested under one 'mood/' folder, so extract_features.py
labels them all as "mood" instead of "mood_Happy", "mood_Melancholic", etc.

This script:
1. Moves mood/Happy/* → mood_Happy/*, mood/Calm/* → mood_Calm/*, etc.
2. Flattens genre_Rock/Rock/* → genre_Rock/* (removes unnecessary nesting)
3. Removes empty leftover folders (genre_Cinematic, genre_Modern, mood_Mellow, etc.)
4. Removes augmented Electronic files (aug_high_*, aug_low_*) to reduce imbalance
"""

import os
import shutil
import glob

BASE = '/Users/macarena.nadeau/Desktop/songpitch-trainer/training_data'

def move_files(src_dir, dest_dir):
    """Move all files from src_dir to dest_dir."""
    if not os.path.exists(src_dir):
        print(f"  ⚠️  Source not found: {src_dir}")
        return 0
    os.makedirs(dest_dir, exist_ok=True)
    count = 0
    for f in os.listdir(src_dir):
        src = os.path.join(src_dir, f)
        if os.path.isfile(src):
            dest = os.path.join(dest_dir, f)
            if not os.path.exists(dest):
                shutil.move(src, dest)
                count += 1
    return count


def fix_structure():
    print("🔧 Fixing training_data folder structure...\n")

    # ── 1. Promote mood subfolders to top level ──
    print("📂 Step 1: Moving mood subfolders to top level...")
    mood_dir = os.path.join(BASE, 'mood')
    if os.path.exists(mood_dir):
        for subfolder in os.listdir(mood_dir):
            sub_path = os.path.join(mood_dir, subfolder)
            if os.path.isdir(sub_path):
                dest = os.path.join(BASE, f'mood_{subfolder}')
                count = move_files(sub_path, dest)
                print(f"  ✅ mood/{subfolder} → mood_{subfolder}: {count} files moved")
    else:
        print("  ⚠️  No mood/ folder found")

    # ── 2. Flatten genre_Rock/Rock/ ──
    print("\n📂 Step 2: Flattening genre_Rock/Rock/...")
    rock_nested = os.path.join(BASE, 'genre_Rock', 'Rock')
    rock_dest = os.path.join(BASE, 'genre_Rock')
    if os.path.exists(rock_nested):
        count = move_files(rock_nested, rock_dest)
        print(f"  ✅ genre_Rock/Rock/ → genre_Rock/: {count} files moved")
    else:
        print("  ⚠️  No nested Rock/ subfolder found")

    # ── 3. Remove empty/leftover folders ──
    print("\n🗑️  Step 3: Removing empty folders...")
    empty_candidates = [
        'mood',           # now empty after step 1
        'genre_Cinematic',
        'genre_Modern',
        'mood_Mellow',
        'mood_Upbeat',
        'mood_Subdued',
    ]
    for folder in empty_candidates:
        path = os.path.join(BASE, folder)
        if os.path.exists(path):
            # Remove folder and any empty subdirs
            try:
                shutil.rmtree(path)
                print(f"  ✅ Removed: {folder}/")
            except Exception as e:
                print(f"  ⚠️  Could not remove {folder}/: {e}")

    # Also remove empty nested Rock subfolder
    rock_empty = os.path.join(BASE, 'genre_Rock', 'Rock')
    if os.path.exists(rock_empty) and not os.listdir(rock_empty):
        os.rmdir(rock_empty)
        print("  ✅ Removed empty: genre_Rock/Rock/")

    # ── 4. Cap Electronic by removing augmented files ──
    print("\n⚡ Step 4: Capping genre_Electronic (removing augmented files)...")
    electronic_dir = os.path.join(BASE, 'genre_Electronic')
    if os.path.exists(electronic_dir):
        aug_files = glob.glob(os.path.join(electronic_dir, 'aug_high_*')) + \
                    glob.glob(os.path.join(electronic_dir, 'aug_low_*'))
        for f in aug_files:
            os.remove(f)
        print(f"  ✅ Removed {len(aug_files)} augmented files from genre_Electronic")

    # ── 5. Print final summary ──
    print("\n📊 Final training_data structure:")
    total = 0
    for folder in sorted(os.listdir(BASE)):
        path = os.path.join(BASE, folder)
        if os.path.isdir(path):
            count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            status = "✅" if count >= 50 else "⚠️ " if count > 0 else "❌"
            print(f"  {status} {folder}: {count} files")
            total += count
    print(f"\n  Total: {total} files")


if __name__ == '__main__':
    fix_structure()
