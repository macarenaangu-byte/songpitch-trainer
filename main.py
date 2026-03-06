from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import pickle
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional

load_dotenv()

# Rate limiter — keyed by client IP
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS for the dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── File upload validation constants ─────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 50
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_AUDIO_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave",
    "audio/flac", "audio/x-flac", "audio/ogg", "audio/aac", "audio/mp4",
    "audio/x-m4a", "audio/m4a",
}
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".mp4"}

def validate_audio_upload(file: UploadFile):
    """Validate uploaded file is an allowed audio format and within size limits."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
        )
    if file.content_type and file.content_type not in ALLOWED_AUDIO_TYPES and file.content_type != "application/octet-stream":
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type '{file.content_type}'. Upload an audio file."
        )

# 1. GLOBAL VARIABLES FOR AI MODELS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yamnet_model = None
genre_model = None
mood_model = None
gen_encoder = None
mood_encoder = None

def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(y_true_int, depth=num_classes)
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = alpha * y_true_one_hot * tf.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)
    return loss_fn

# 🔥 THIS IS THE FIX: Load models AFTER the server port opens
@app.on_event("startup")
async def load_all_models():
    global yamnet_model, genre_model, mood_model, gen_encoder, mood_encoder
    print("🚪 Port is open! Now loading AI brains in the background...")
    
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    genre_model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, 'yamnet_genre_model.keras'),
        custom_objects={'loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
    )
    mood_model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, 'yamnet_mood_model.keras'),
        custom_objects={'loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
    )
    
    with open(os.path.join(BASE_DIR, 'yamnet_genre_model_encoder.pkl'), 'rb') as f:
        gen_encoder = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'yamnet_mood_model_encoder.pkl'), 'rb') as f:
        mood_encoder = pickle.load(f)
        
    print("✅ All AI Brains successfully loaded and ready for traffic!")

# OpenAI client for AI Brief Writer
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BRIEF_SYSTEM_PROMPT = """You are a music industry brief writer for SongPitch, a platform connecting composers with music executives.

Given rough notes from a music executive, generate a polished opportunity description and suggest appropriate genres, moods, and project type.

AVAILABLE GENRES (use ONLY these exact names):
Classical, Jazz, Electronic, Hip-Hop, Pop, Film Score, Ambient, R&B, Afrobeats, World Music, Musical Theatre, Rock, Country, Folk, Blues, Reggae, Latin, K-Pop, EDM, Indie

AVAILABLE MOODS (use ONLY these exact names):
Uplifting, Melancholic, Energetic, Calm, Dark, Romantic, Epic, Playful, Aggressive, Dreamy, Nostalgic, Mysterious, Triumphant, Tense

PROJECT TYPES (use ONLY these exact names):
Film, TV Series, Advertising, Trailer, Video Game, Podcast, Social Media, Other

RULES:
1. Write a professional, engaging description (2-4 sentences) based on the notes
2. Select 1-3 genres that best fit
3. Select 1-3 moods that best fit
4. Select exactly 1 project type
5. Return ONLY valid JSON with no extra text

Return format:
{
  "description": "...",
  "genres": ["...", "..."],
  "moods": ["...", "..."],
  "project_type": "..."
}"""

ALLOWED_GENRES = {"Classical", "Jazz", "Electronic", "Hip-Hop", "Pop", "Film Score", "Ambient", "R&B", "Afrobeats", "World Music", "Musical Theatre", "Rock", "Country", "Folk", "Blues", "Reggae", "Latin", "K-Pop", "EDM", "Indie"}
ALLOWED_MOODS = {"Uplifting", "Melancholic", "Energetic", "Calm", "Dark", "Romantic", "Epic", "Playful", "Aggressive", "Dreamy", "Nostalgic", "Mysterious", "Triumphant", "Tense"}

class BriefRequest(BaseModel):
    notes: str
    title: Optional[str] = None
    project_type: Optional[str] = None

@app.post("/predict")
# 🔥 THIS IS THE FIX: Disabled limiter to prevent Typing crash
# @limiter.limit("10/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    if yamnet_model is None:
        raise HTTPException(status_code=503, detail="AI is still warming up. Try again in 30 seconds!")
        
    validate_audio_upload(file)
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE_MB}MB.")
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(contents)

    try:
        y, sr = librosa.load(temp_path, sr=16000, duration=30)
        target_length = 30 * 16000
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]

        y = y.astype(np.float32)
        if np.abs(y).max() > 0:
            y = y / max(np.abs(y).max(), 1.0)

        scores, embeddings, spectrogram = yamnet_model(y)
        embedding = np.mean(embeddings.numpy(), axis=0)  
        X = embedding[np.newaxis, :]  

        gen_preds = genre_model.predict(X)
        mood_preds = mood_model.predict(X)

        top2_gen = np.argsort(gen_preds[0])[-2:][::-1]
        primary_genre = str(gen_encoder.inverse_transform([top2_gen[0]])[0])
        secondary_genre = str(gen_encoder.inverse_transform([top2_gen[1]])[0])
        primary_genre_conf = float(gen_preds[0][top2_gen[0]])
        secondary_genre_conf = float(gen_preds[0][top2_gen[1]])

        best_mood_idx = np.argmax(mood_preds[0])
        mood_result = str(mood_encoder.inverse_transform([best_mood_idx])[0])
        mood_conf = float(mood_preds[0][best_mood_idx])

        def clean_label(label):
            return label.replace('genre_', '').replace('mood_', '').replace('_', ' ').upper()

        clean_genre = clean_label(primary_genre)
        clean_secondary = clean_label(secondary_genre)
        clean_mood = clean_label(mood_result)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            "predictions": [clean_genre, clean_mood],
            "genre": clean_genre,
            "genre_confidence": primary_genre_conf,
            "secondary_genre": clean_secondary,
            "secondary_genre_confidence": secondary_genre_conf,
            "mood": clean_mood,
            "mood_confidence": mood_conf,
        }

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return {"error": str(e)}

@app.post("/transcribe")
# @limiter.limit("5/minute")
async def transcribe(request: Request, file: UploadFile = File(...)):
    validate_audio_upload(file)
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE_MB}MB.")
    temp_path = f"temp_transcribe_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(contents)

    try:
        with open(temp_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
            )
        lyrics = transcript.strip() if transcript else ""
        if len(lyrics) < 10 or lyrics.lower() in ["", "you", "thank you", "thanks for watching"]:
            lyrics = ""
        return {"status": "success", "lyrics": lyrics}
    except Exception as e:
        return {"status": "error", "lyrics": "", "message": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/generate-brief")
# @limiter.limit("15/minute")
async def generate_brief(request: Request, req: BriefRequest):
    if not req.notes or not req.notes.strip():
        return {"status": "error", "message": "Please provide some notes to generate a brief."}

    user_message = f"Notes: {req.notes.strip()}"
    if req.title:
        user_message = f"Title: {req.title}\n{user_message}"
    if req.project_type:
        user_message += f"\nProject type hint: {req.project_type}"

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": BRIEF_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=500,
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        result = json.loads(content)

        result["genres"] = [g for g in result.get("genres", []) if g in ALLOWED_GENRES]
        result["moods"] = [m for m in result.get("moods", []) if m in ALLOWED_MOODS]

        return {"status": "success", **result}

    except json.JSONDecodeError:
        return {"status": "error", "message": "AI returned invalid format. Please try again."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)