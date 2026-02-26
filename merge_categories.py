import os
import shutil

# Define the "Master Categories"
# We are grouping similar textures to help the AI stop guessing
MERGE_MAP = {
    'mood_Mellow': ['mood_Calm', 'mood_Atmospheric', 'mood_Suspense'],
    'mood_Upbeat': ['mood_Happy', 'mood_Energetic'],
    'genre_Cinematic': ['genre_Film Score', 'genre_Orchestral'],
    'genre_Modern': ['genre_Pop', 'genre_Electronic']
}

BASE_PATH = 'training_data' # Change this to your Desktop folder path if needed

def merge_folders():
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
    merge_folders()
    print("✨ Merging complete! Your dataset is now more balanced.")