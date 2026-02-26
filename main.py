import os
import json
import shutil
import pickle
import librosa
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load dual models ────────────────────────────────────────────────────────
print("📂 Loading models...")

genre_model = tf.keras.models.load_model('genre_model.h5')
with open('genre_model_encoder.pkl', 'rb') as f:
    genre_encoder = pickle.load(f)

mood_model = tf.keras.models.load_model('mood_model.h5')
with open('mood_model_encoder.pkl', 'rb') as f:
    mood_encoder = pickle.load(f)

print(f"✅ Genre model: {len(genre_encoder.classes_)} classes — {list(genre_encoder.classes_)}")
print(f"✅ Mood model:  {len(mood_encoder.classes_)} classes — {list(mood_encoder.classes_)}")

# Load exact normalization params from training (must match training preprocessing)
with open('norm_params.pkl', 'rb') as f:
    norm = pickle.load(f)
NORM_MIN = norm['X_min']
NORM_MAX = norm['X_max']
print(f"📏 Normalization: min={NORM_MIN}, max={NORM_MAX}")


def preprocess_audio(file_path):
    """Load audio, generate mel spectrogram, preprocess for MobileNetV2."""
    y, sr = librosa.load(file_path, sr=22050, duration=30)
    target_len = 30 * 22050
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Shape for MobileNetV2: (1, 128, 128, 3)
    X = mel_spec_db[np.newaxis, ..., np.newaxis]
    X = np.repeat(X, 3, axis=-1)
    X = tf.image.resize(X, [128, 128]).numpy()

    # Normalize to [0, 255] then apply MobileNetV2 preprocessing (→ [-1, 1])
    X = (X - NORM_MIN) / (NORM_MAX - NORM_MIN) * 255.0
    X = np.clip(X, 0, 255)
    X = tf.keras.applications.mobilenet_v2.preprocess_input(X)

    return X


@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        X = preprocess_audio(temp_path)

        # Genre prediction — top 3
        genre_preds = genre_model.predict(X, verbose=0)[0]
        genre_top = np.argsort(genre_preds)[-3:][::-1]
        genre_labels = [str(genre_encoder.inverse_transform([i])[0]) for i in genre_top]
        genre_confidences = [float(genre_preds[i]) for i in genre_top]

        # Mood prediction — top 3
        mood_preds = mood_model.predict(X, verbose=0)[0]
        mood_top = np.argsort(mood_preds)[-3:][::-1]
        mood_labels = [str(mood_encoder.inverse_transform([i])[0]) for i in mood_top]
        mood_confidences = [float(mood_preds[i]) for i in mood_top]

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            "status": "success",
            "genre": genre_labels[0],
            "mood": mood_labels[0],
            "genre_confidence": genre_confidences[0],
            "mood_confidence": mood_confidences[0],
            "genre_top3": [{"label": l, "confidence": c} for l, c in zip(genre_labels, genre_confidences)],
            "mood_top3": [{"label": l, "confidence": c} for l, c in zip(mood_labels, mood_confidences)],
            # Backward compatibility — combined list of top predictions
            "predictions": genre_labels + mood_labels,
        }

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


# ─── AI Brief Writer ─────────────────────────────────────────────────────────
# Generates polished opportunity descriptions from rough notes using GPT-4o-mini

openai_client = None
if os.getenv("OPENAI_API_KEY"):
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("✅ OpenAI client initialized for AI Brief Writer")
else:
    print("⚠️  No OPENAI_API_KEY found — AI Brief Writer disabled")

BRIEF_SYSTEM_PROMPT = """You are a music industry brief writer for SongPitch, a professional platform connecting composers with music executives (film, TV, advertising, games).

Your job: take rough notes from a music executive and generate a polished, professional opportunity description that will attract the right composers.

AVAILABLE GENRES (use ONLY these exact names):
Classical, Jazz, Electronic, Hip-Hop, Pop, Film Score, Ambient, R&B, Afrobeats, World Music, Musical Theatre, Rock, Country, Folk, Blues, Reggae, Latin, K-Pop, EDM, Indie

AVAILABLE MOODS (use ONLY these exact names):
Uplifting, Melancholic, Energetic, Calm, Dark, Romantic, Epic, Playful, Aggressive, Dreamy, Nostalgic, Mysterious, Triumphant, Tense

PROJECT TYPES (use ONLY these exact names):
Film, TV Series, Advertising, Trailer, Video Game, Podcast, Social Media, Other

INSTRUCTIONS:
1. Write a 2-4 sentence professional description based on the user's rough notes
2. The description should be clear, specific, and attractive to composers
3. Include key details: style, tempo feel, instrumentation hints, reference points
4. Suggest 1-3 genres and 1-3 moods that best match the project
5. Suggest the most appropriate project type

Return ONLY valid JSON (no markdown, no code fences):
{
  "description": "Professional 2-4 sentence description",
  "genres": ["Genre1", "Genre2"],
  "moods": ["Mood1", "Mood2"],
  "project_type": "ProjectType"
}"""


class BriefRequest(BaseModel):
    notes: str
    title: Optional[str] = None
    project_type: Optional[str] = None


@app.post("/generate-brief")
async def generate_brief(req: BriefRequest):
    if not openai_client:
        raise HTTPException(status_code=503, detail="AI Brief Writer not configured — missing OPENAI_API_KEY")

    if not req.notes or len(req.notes.strip()) < 10:
        raise HTTPException(status_code=400, detail="Please provide at least a short description of your project")

    # Build user prompt with optional context
    user_msg = req.notes
    if req.title:
        user_msg = f"Project title: {req.title}\n\n{user_msg}"
    if req.project_type:
        user_msg = f"{user_msg}\n\nProject type: {req.project_type}"

    try:
        print(f"✨ Generating brief from: {req.notes[:80]}...")

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": BRIEF_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=500,
        )

        raw = response.choices[0].message.content.strip()

        # Parse JSON response (handle possible markdown code fences)
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        result = json.loads(raw)

        # Validate genres/moods are from our allowed lists
        valid_genres = {'Classical', 'Jazz', 'Electronic', 'Hip-Hop', 'Pop', 'Film Score',
                        'Ambient', 'R&B', 'Afrobeats', 'World Music', 'Musical Theatre',
                        'Rock', 'Country', 'Folk', 'Blues', 'Reggae', 'Latin', 'K-Pop', 'EDM', 'Indie'}
        valid_moods = {'Uplifting', 'Melancholic', 'Energetic', 'Calm', 'Dark', 'Romantic',
                       'Epic', 'Playful', 'Aggressive', 'Dreamy', 'Nostalgic', 'Mysterious',
                       'Triumphant', 'Tense'}
        valid_types = {'Film', 'TV Series', 'Advertising', 'Trailer', 'Video Game',
                       'Podcast', 'Social Media', 'Other'}

        result['genres'] = [g for g in result.get('genres', []) if g in valid_genres][:3]
        result['moods'] = [m for m in result.get('moods', []) if m in valid_moods][:3]
        if result.get('project_type') not in valid_types:
            result['project_type'] = 'Other'

        print(f"✅ Brief generated — genres: {result['genres']}, moods: {result['moods']}")
        return {"status": "success", **result}

    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parse error: {e}\nRaw response: {raw}")
        raise HTTPException(status_code=500, detail="AI returned invalid format — please try again")
    except Exception as e:
        print(f"❌ Brief generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
