from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
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
yamnet_model   = None
mood_model     = None
mood_encoder   = None

# Hierarchical genre models
stage1_model   = None   # broad category classifier (10 categories)
stage1_encoder = None
stage2_models  = {}     # {category_name: Keras model}
stage2_encoders = {}    # {category_name: LabelEncoder}

STAGE2_CATEGORIES = [
    'Latin', 'Electronic', 'Rock_Metal', 'Classical_Cinematic',
    'HipHop_Urban', 'Pop_Indie', 'Folk_Country_Roots', 'Jazz_Blues',
    'Ambient_Chill', 'Theatrical',
]

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
    global yamnet_model, mood_model, mood_encoder
    global stage1_model, stage1_encoder, stage2_models, stage2_encoders
    print("🚪 Port is open! Now loading AI brains in the background...")

    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

    # ── Mood model (unchanged) ──
    mood_model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, 'yamnet_mood_model.h5'),
        custom_objects={'loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
    )
    with open(os.path.join(BASE_DIR, 'yamnet_mood_model_encoder.pkl'), 'rb') as f:
        mood_encoder = pickle.load(f)

    # ── Hierarchical genre models ──
    stage1_model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, 'stage1_model.h5'),
        custom_objects={'loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
    )
    with open(os.path.join(BASE_DIR, 'stage1_model_encoder.pkl'), 'rb') as f:
        stage1_encoder = pickle.load(f)

    for cat in STAGE2_CATEGORIES:
        model_path   = os.path.join(BASE_DIR, f'stage2_{cat}_model.h5')
        encoder_path = os.path.join(BASE_DIR, f'stage2_{cat}_model_encoder.pkl')
        if os.path.exists(model_path):
            stage2_models[cat] = tf.keras.models.load_model(
                model_path,
                custom_objects={'loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
            )
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                stage2_encoders[cat] = pickle.load(f)

    loaded_s2 = sum(1 for cat in STAGE2_CATEGORIES if cat in stage2_models)
    print(f"✅ Hierarchical genre: Stage 1 + {loaded_s2}/{len(STAGE2_CATEGORIES)} Stage 2 models loaded")
    print("✅ All AI Brains successfully loaded and ready for traffic!")

# OpenAI client for AI Brief Writer
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BRIEF_SYSTEM_PROMPT = """You are a music industry brief writer for SongPitch, a platform connecting composers with music executives.

Given rough notes from a music executive, generate a polished opportunity description and suggest appropriate genres, moods, and project type.

AVAILABLE GENRES (use ONLY these exact names):
Classical, Jazz, Electronic, Hip-Hop, Pop, Film Score, Ambient, R&B, Afrobeats, World Music, Musical Theatre, Rock, Country, Folk, Blues, Reggae, Latin, K-Pop, EDM, Indie, Gospel, Lo-Fi, Corporate, Cinematic, Children's, Funk/Soul, Trap, New Age, Acoustic, House, Metal, Bachata, Cumbia, Merengue, Tango, Flamenco, Trap Latino, Reggaetón, Dancehall, Techno, Trance, Drum & Bass, Dubstep, Synthwave, Punk, Hard Rock, Alternative Rock, Grunge, Progressive Rock, Opera, Baroque, HyperPop, Urbano

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

ALLOWED_GENRES = {"Classical", "Jazz", "Electronic", "Hip-Hop", "Pop", "Film Score", "Ambient", "R&B", "Afrobeats", "World Music", "Musical Theatre", "Rock", "Country", "Folk", "Blues", "Reggae", "Latin", "K-Pop", "EDM", "Indie", "Gospel", "Lo-Fi", "Corporate", "Cinematic", "Children's", "Funk/Soul", "Trap", "New Age", "Acoustic", "House", "Metal", "Bachata", "Cumbia", "Merengue", "Tango", "Flamenco", "Trap Latino", "Reggaetón", "Dancehall", "Techno", "Trance", "Drum & Bass", "Dubstep", "Synthwave", "Punk", "Hard Rock", "Alternative Rock", "Grunge", "Progressive Rock", "Opera", "Baroque", "HyperPop", "Urbano"}
ALLOWED_MOODS = {"Uplifting", "Melancholic", "Energetic", "Calm", "Dark", "Romantic", "Epic", "Playful", "Aggressive", "Dreamy", "Nostalgic", "Mysterious", "Triumphant", "Tense"}

class BriefRequest(BaseModel):
    notes: str
    title: Optional[str] = None
    project_type: Optional[str] = None

@app.post("/predict")
# 🔥 THIS IS THE FIX: Disabled limiter to prevent Typing crash
# @limiter.limit("10/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    if yamnet_model is None or stage1_model is None:
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

        mood_preds = mood_model.predict(X)

        # ── Two-stage genre prediction ──────────────────────────────────────
        # Stage 1: broad category
        cat_preds        = stage1_model.predict(X)
        top2_cat         = np.argsort(cat_preds[0])[-2:][::-1]
        predicted_cat    = str(stage1_encoder.inverse_transform([top2_cat[0]])[0])
        runner_up_cat    = str(stage1_encoder.inverse_transform([top2_cat[1]])[0])
        stage1_conf      = float(cat_preds[0][top2_cat[0]])

        # Stage 2: specific genre within the predicted category
        s2_model   = stage2_models.get(predicted_cat)
        s2_encoder = stage2_encoders.get(predicted_cat)

        if s2_model is not None and s2_encoder is not None:
            s2_preds = s2_model.predict(X)
            top2_gen = np.argsort(s2_preds[0])[-2:][::-1]
            primary_genre        = str(s2_encoder.inverse_transform([top2_gen[0]])[0])
            secondary_genre      = str(s2_encoder.inverse_transform([top2_gen[1]])[0])
            primary_genre_conf   = float(s2_preds[0][top2_gen[0]]) * stage1_conf
            secondary_genre_conf = float(s2_preds[0][top2_gen[1]]) * stage1_conf
        elif s2_encoder is not None:
            # Single-genre category: trivially correct, return the only genre
            primary_genre        = str(s2_encoder.classes_[0])
            secondary_genre      = primary_genre
            primary_genre_conf   = stage1_conf
            secondary_genre_conf = stage1_conf
        else:
            # Fallback: use runner-up category's top genre if stage2 model unavailable
            s2_fb_model   = stage2_models.get(runner_up_cat)
            s2_fb_encoder = stage2_encoders.get(runner_up_cat)
            if s2_fb_model is not None and s2_fb_encoder is not None:
                s2_fb_preds  = s2_fb_model.predict(X)
                fb_top_idx   = int(np.argmax(s2_fb_preds[0]))
                primary_genre        = str(s2_fb_encoder.inverse_transform([fb_top_idx])[0])
                secondary_genre      = primary_genre
                primary_genre_conf   = float(cat_preds[0][top2_cat[1]])
                secondary_genre_conf = primary_genre_conf
            else:
                primary_genre        = f"genre_{predicted_cat}"
                secondary_genre      = primary_genre
                primary_genre_conf   = stage1_conf
                secondary_genre_conf = stage1_conf

        best_mood_idx = np.argmax(mood_preds[0])
        mood_result = str(mood_encoder.inverse_transform([best_mood_idx])[0])
        mood_conf = float(mood_preds[0][best_mood_idx])

        # Map raw model labels → exact GENRE_OPTIONS / MOOD_OPTIONS display strings.
        # "Orchestral" maps to "Classical" — closest app genre to the training label.
        # "Alternative" maps to "Indie" — not a separate option in the app.
        # All others map 1:1 with correct Title Case so they match <select> option values.
        GENRE_MAP = {
            # Core genre labels (folder name → display name)
            "classical":         "Classical",
            "jazz":              "Jazz",
            "electronic":        "Electronic",
            "hip-hop":           "Hip-Hop",
            "hip_hop":           "Hip-Hop",
            "pop":               "Pop",
            "film_score":        "Film Score",
            "ambient":           "Ambient",
            "r&b":               "R&B",
            "afrobeats":         "Afrobeats",
            "world_music":       "World Music",
            "musical_theatre":   "Musical Theatre",
            "rock":              "Rock",
            "country":           "Country",
            "folk":              "Folk",
            "blues":             "Blues",
            "reggae":            "Reggae",
            "latin":             "Latin",
            "k-pop":             "K-Pop",
            "kpop":              "K-Pop",
            "edm":               "EDM",
            "indie":             "Indie",
            "gospel":            "Gospel",
            "lo_fi":             "Lo-Fi",
            "lo-fi":             "Lo-Fi",
            "corporate":         "Corporate",
            "cinematic":         "Cinematic",
            "childrens":         "Children's",
            # Phase 2 new genres
            "funk_soul":         "Funk/Soul",
            "funk/soul":         "Funk/Soul",
            "trap":              "Trap",
            "new_age":           "New Age",
            "acoustic":          "Acoustic",
            "house":             "House",
            "metal":             "Metal",
            # Phase 3 — Latin sub-genres
            "bachata":           "Bachata",
            "cumbia":            "Cumbia",
            "merengue":          "Merengue",
            "tango":             "Tango",
            "flamenco":          "Flamenco",
            "trap_latino":       "Trap Latino",
            "trap-latino":       "Trap Latino",
            "reggaeton":         "Reggaetón",
            "reggaetón":         "Reggaetón",
            "dancehall":         "Dancehall",
            # Phase 3 — Electronic sub-genres
            "techno":            "Techno",
            "trance":            "Trance",
            "drum_and_bass":     "Drum & Bass",
            "drum-and-bass":     "Drum & Bass",
            "dubstep":           "Dubstep",
            "synthwave":         "Synthwave",
            # Phase 3 — Rock / Alt sub-genres
            "punk":              "Punk",
            "hard_rock":         "Hard Rock",
            "hard-rock":         "Hard Rock",
            "alternative_rock":  "Alternative Rock",
            "alternative-rock":  "Alternative Rock",
            "grunge":            "Grunge",
            "progressive_rock":  "Progressive Rock",
            "progressive-rock":  "Progressive Rock",
            # Phase 3 — Classical sub-genres
            "opera":             "Opera",
            "baroque":           "Baroque",
            # Phase 3 — Urban
            "hyperpop":          "HyperPop",
            "urbano":            "Urbano",
            # Legacy labels kept for backwards compatibility
            "orchestral":        "Classical",
            "alternative":       "Indie",
        }
        MOOD_MAP = {
            "aggressive":   "Aggressive",
            "atmospheric":  "Dreamy",
            "calm":         "Calm",
            "dark":         "Dark",
            "energetic":    "Energetic",
            "epic":         "Epic",
            "happy":        "Uplifting",
            "melancholic":  "Melancholic",
            "mysterious":   "Mysterious",
            "nostalgic":    "Nostalgic",
            "playful":      "Playful",
            "triumphant":   "Triumphant",
            "romantic":     "Romantic",
            "tense":        "Tense",
            "uplifting":    "Uplifting",
            "dreamy":       "Dreamy",
        }

        def normalize_genre(label):
            raw = label.replace('genre_', '').replace('_', '-').lower().replace('-', '_')
            # Try both hyphen and underscore variants
            key = label.replace('genre_', '').replace('_', '-').lower()
            key_under = label.replace('genre_', '').lower()
            return GENRE_MAP.get(key_under) or GENRE_MAP.get(key) or label.replace('genre_', '').replace('_', ' ').title()

        def normalize_mood(label):
            key = label.replace('mood_', '').lower()
            return MOOD_MAP.get(key) or label.replace('mood_', '').replace('_', ' ').title()

        clean_genre = normalize_genre(primary_genre)
        clean_secondary = normalize_genre(secondary_genre)
        clean_mood = normalize_mood(mood_result)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            "predictions": [clean_genre, clean_mood],
            "genre": clean_genre,
            "genre_confidence": primary_genre_conf,
            "secondary_genre": clean_secondary,        # snake_case (legacy)
            "secondaryGenre": clean_secondary,         # camelCase (used by PortfolioPage bulk upload)
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
                temperature=0.0, # 👈 Hardcoded so the AI is literal
                prompt="This is a music track. If there are no clear human vocals, return an empty string. Do not invent subtitles, do not output watermarks, and do not use emojis." # 👈 Hardcoded guardrail
            )
        lyrics = transcript.strip() if transcript else ""
        
        # 🛡️ The Final Blacklist Shield
        hallucination_blacklist = ["you", "thank you", "thanks for watching", "sous-titrage", "subtitles", "amara.org", "mr beast", "hodori"]
        if len(lyrics) < 10 or any(bad_word in lyrics.lower() for bad_word in hallucination_blacklist):
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