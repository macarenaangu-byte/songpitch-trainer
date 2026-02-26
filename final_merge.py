import os
import shutil

# Merging the hardest-to-distinguish moods into one super-category
MERGE_MAP = {
    'mood_Subdued': ['mood_Calm', 'mood_Melancholic', 'mood_Atmospheric']
}

BASE_PATH = 'training_data'

def final_cleanup():
    for master, children in MERGE_MAP.items():
        master_path = os.path.join(BASE_PATH, master)
        os.makedirs(master_path, exist_ok=True)
        for child in children:
            child_path = os.path.join(BASE_PATH, child)
            if os.path.exists(child_path):
                print(f"📦 Merging {child} into {master}...")
                for file in os.listdir(child_path):
                    shutil.move(os.path.join(child_path, file), os.path.join(master_path, file))
                os.rmdir(child_path)

if __name__ == "__main__":
    final_cleanup()
    print("✅ Moods refined! Now you have a powerful 'Subdued' category.")