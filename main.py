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
yamnet_model       = None
mood_model         = None
mood_encoder       = None
instrument_model   = None
instrument_encoder = None   # list of class names
feature_scaler     = None   # StandardScaler fitted on training set

# Hierarchical genre models
stage1_model    = None   # broad category classifier (10 categories)
stage1_encoder  = None
stage2_models   = {}     # {category_name: Keras model}
stage2_encoders = {}     # {category_name: LabelEncoder}

STAGE2_CATEGORIES = [
    'Latin', 'Electronic', 'Rock_Metal', 'Classical_Cinematic',
    'HipHop_Urban', 'Pop_Indie', 'Folk_Country_Roots', 'Jazz_Blues',
    'Ambient_Chill', 'Theatrical',
]

# Discogs TF1 frozen graph (loaded at startup)
discogs_session      = None
discogs_input_tensor = None
discogs_embed_tensor = None

# Librosa constants — must match extract_discogs_features.py exactly
SR_LIBROSA  = 22050
SR_DISCOGS  = 16000
SR_YAMNET   = 16000   # kept for mood model (YAMNet still used for mood)
N_MFCC      = 20
HOP_LENGTH  = 512

# Discogs-EfficientNet mel-patch constants — must match extract_discogs_features.py exactly
N_FFT_D      = 512
HOP_LEN_D    = 256
N_MELS_D     = 96
FMIN_D       = 0
FMAX_D       = 8000
PATCH_FRAMES = 128
PATCH_HOP    = 64
BATCH_SIZE_D = 64

# ─── YAMNet vocal class indices (from AudioSet ontology) ─────────────────────
_YAMNET_SINGING     = 24   # "Singing"
_YAMNET_CHOIR       = 25   # "Choir"
_YAMNET_RAPPING     = 31   # "Rapping"
_YAMNET_VOCAL_MUSIC = 249  # "Vocal music"
_YAMNET_SPEECH      = 0    # "Speech"

def detect_vocals(yamnet_class_scores: np.ndarray) -> tuple[str, float]:
    """Detect vocal presence from YAMNet class score matrix.

    Args:
        yamnet_class_scores: (N_frames, 521) raw YAMNet class scores per frame

    Returns:
        (vocal_type, confidence)
        vocal_type: "Instrumental" | "Vocals" | "Choir" | "Rap / Spoken Word"
    """
    mean_scores = np.mean(yamnet_class_scores, axis=0)   # (521,)

    singing_score = float(mean_scores[_YAMNET_SINGING])
    choir_score   = float(mean_scores[_YAMNET_CHOIR])
    rap_score     = float(mean_scores[_YAMNET_RAPPING])
    vocal_score   = float(mean_scores[_YAMNET_VOCAL_MUSIC])
    speech_score  = float(mean_scores[_YAMNET_SPEECH])

    # Combined vocal presence: any singing / rap / spoken word signal
    total_vocal = max(singing_score, vocal_score, choir_score * 0.8, rap_score * 0.8)

    VOCAL_THRESHOLD = 0.08   # tuned empirically — below this → instrumental

    if total_vocal < VOCAL_THRESHOLD:
        return "Instrumental", round(1.0 - total_vocal, 3)
    if choir_score > singing_score and choir_score > 0.05:
        return "Choir", round(choir_score, 3)
    if rap_score > singing_score and rap_score > 0.05:
        return "Rap / Spoken Word", round(rap_score, 3)
    return "Vocals", round(max(singing_score, vocal_score), 3)


def get_tempo_descriptor(bpm: int) -> str:
    """Map BPM to a human-readable tempo descriptor used by sync licensing platforms."""
    if bpm < 60:   return "Very Slow"
    if bpm < 80:   return "Slow"
    if bpm < 100:  return "Moderate"
    if bpm < 120:  return "Upbeat"
    if bpm < 140:  return "Fast"
    return "Very Fast"


def get_use_cases(genre: str, moods: list, energy: int, vocals: str) -> list[str]:
    """Derive sync licensing use case tags from genre + mood + energy + vocal type.

    No model needed — pure rule-based mapping used by all major catalog platforms.
    Returns up to 4 use case strings ordered by confidence.
    """
    g = genre.lower()
    all_moods = {m["mood"].lower() for m in moods}
    tags = []

    # ── Film & TV ─────────────────────────────────────────────────────────────
    film_genres = {'film score', 'cinematic', 'classical', 'baroque', 'opera', 'ambient'}
    film_moods  = {'epic', 'mysterious', 'suspense', 'dark', 'triumphant', 'tense', 'atmospheric'}
    if g in film_genres or len(all_moods & film_moods) >= 1:
        tags.append('Film & TV')

    # ── Trailer / Epic ────────────────────────────────────────────────────────
    if ('epic' in all_moods or 'triumphant' in all_moods) and energy >= 6:
        tags.append('Trailer / Epic')

    # ── Advertising / Commercial ──────────────────────────────────────────────
    ad_moods = {'uplifting', 'happy', 'energetic', 'inspiring', 'playful', 'triumphant'}
    if len(all_moods & ad_moods) >= 1 and energy >= 4:
        tags.append('Advertising')

    # ── Corporate / Brand ─────────────────────────────────────────────────────
    corp_genres = {'corporate', 'acoustic', 'pop', 'indie'}
    corp_moods  = {'uplifting', 'happy', 'inspiring', 'calm', 'playful'}
    if g in corp_genres or len(all_moods & corp_moods) >= 1:
        tags.append('Corporate / Brand')

    # ── Sports / Action ───────────────────────────────────────────────────────
    if energy >= 8 or (energy >= 7 and len(all_moods & {'aggressive', 'energetic'}) >= 1):
        tags.append('Sports / Action')

    # ── Gaming ────────────────────────────────────────────────────────────────
    game_genres = {'electronic', 'techno', 'dubstep', 'drum & bass', 'edm', 'trance',
                   'house', 'metal', 'rock', 'synthwave'}
    game_moods  = {'epic', 'aggressive', 'mysterious', 'dark', 'energetic'}
    if (g in game_genres and energy >= 6) or len(all_moods & game_moods) >= 2:
        tags.append('Gaming')

    # ── Study / Focus / Background ────────────────────────────────────────────
    chill_genres = {'lo-fi', 'ambient', 'new age', 'corporate', 'classical'}
    if g in chill_genres or (energy <= 4 and vocals == 'Instrumental'):
        tags.append('Study / Focus')

    # ── Meditation / Wellness ─────────────────────────────────────────────────
    if g in {'ambient', 'new age'} and energy <= 4:
        tags.append('Meditation / Wellness')

    # ── Romance / Wedding ─────────────────────────────────────────────────────
    if len(all_moods & {'romantic', 'nostalgic', 'melancholic'}) >= 1 and energy <= 7:
        tags.append('Romance / Wedding')

    # ── Documentary / Nature ──────────────────────────────────────────────────
    doc_genres = {'folk', 'world music', 'classical', 'ambient', 'acoustic', 'new age'}
    if g in doc_genres or 'atmospheric' in all_moods:
        tags.append('Documentary')

    # ── Social Media / Content Creation ──────────────────────────────────────
    social_genres = {'pop', 'indie', 'hip-hop', 'r&b', 'edm', 'trap', 'afrobeats', 'k-pop'}
    if g in social_genres and energy >= 5:
        tags.append('Social Media / Content')

    # Deduplicate and cap at 4 most relevant
    seen = set()
    result = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            result.append(t)
        if len(result) == 4:
            break
    return result


# ─── LUFS loudness measurement (ITU-R BS.1770 via pyloudnorm) ────────────────
import pyloudnorm as pyln

def compute_lufs(y: np.ndarray, sr: int) -> tuple[float, str]:
    """Return (integrated_lufs, note) for the audio signal.

    Uses the ITU-R BS.1770-4 standard — same algorithm used by Spotify,
    YouTube, Apple Music and broadcast regulators.
    """
    try:
        meter = pyln.Meter(sr)                        # BS.1770 meter at audio SR
        lufs  = float(meter.integrated_loudness(y))
        lufs  = max(lufs, -70.0)                      # clamp silence floor

        if lufs <= -23.0:
            note = "Broadcast-safe (≤ -23 LUFS)"
        elif lufs <= -16.0:
            note = "Streaming-safe (Apple Music: -16 LUFS)"
        elif lufs <= -14.0:
            note = "Streaming-safe (Spotify/YouTube: -14 LUFS)"
        elif lufs <= -9.0:
            note = "Slightly loud — will be turned down on streaming platforms"
        else:
            note = "Very loud — will be significantly turned down on streaming platforms"

        return round(lufs, 1), note
    except Exception:
        return -14.0, "Could not measure"


# ─── Time signature detection ─────────────────────────────────────────────────
def detect_time_signature(y: np.ndarray, sr: int, hop_length: int) -> str:
    """Estimate time signature (4/4, 3/4, 6/8) from beat periodicity.

    Uses the onset strength envelope autocorrelation at 3-beat vs 4-beat
    lags to determine whether the dominant meter is triple or duple.
    Returns '4/4', '3/4', or '6/8'.
    """
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        # Autocorrelation of onset envelope reveals metric periodicity
        ac = librosa.autocorrelate(onset_env, max_size=len(onset_env) // 2)
        # Estimate beat period in frames
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=hop_length
        )
        beat_frames = int(round(sr / (float(tempo) / 60.0) / hop_length))
        if beat_frames < 1:
            return "4/4"

        # Compare autocorrelation energy at 3-beat vs 4-beat periods
        lag3 = beat_frames * 3
        lag4 = beat_frames * 4
        lag6 = beat_frames * 6

        def ac_energy(lag, width=3):
            lo = max(0, lag - width)
            hi = min(len(ac), lag + width + 1)
            return float(np.mean(ac[lo:hi])) if hi > lo else 0.0

        e3 = ac_energy(lag3)
        e4 = ac_energy(lag4)
        e6 = ac_energy(lag6)

        if e3 > e4 * 1.15 and e6 > e4 * 0.9:
            return "6/8"
        if e3 > e4 * 1.1:
            return "3/4"
        return "4/4"
    except Exception:
        return "4/4"


# ─── Musical key detection (Krumhansl-Schmuckler profiles) ───────────────────
# These are the classic tonal hierarchy profiles from music cognition research.
# Correlating the song's chromagram against these 24 templates (12 major + 12 minor)
# gives ~80-85% key accuracy — comparable to Essentia's KeyExtractor.
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                       2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                       2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
               'F#', 'G', 'G#', 'A', 'A#', 'B']

def detect_key(chroma: np.ndarray) -> tuple[str, float]:
    """Return (key_string, confidence) using Krumhansl-Schmuckler correlation.

    Args:
        chroma: (12, T) chromagram from librosa.feature.chroma_stft

    Returns:
        key_string like 'C Major' or 'F# Minor'
        confidence in [0, 1] — normalised correlation strength
    """
    chroma_mean = np.mean(chroma, axis=1)          # (12,) — average pitch class energy
    chroma_mean = chroma_mean / (chroma_mean.sum() + 1e-8)  # normalise to sum = 1

    best_corr  = -np.inf
    best_key   = 'C Major'

    for root in range(12):
        # Rotate the profile so it starts on this root note
        maj_profile = np.roll(_KS_MAJOR, root)
        min_profile = np.roll(_KS_MINOR, root)

        # Pearson correlation between chromagram and key profile
        corr_maj = float(np.corrcoef(chroma_mean, maj_profile)[0, 1])
        corr_min = float(np.corrcoef(chroma_mean, min_profile)[0, 1])

        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key  = f'{_NOTE_NAMES[root]} Major'
        if corr_min > best_corr:
            best_corr = corr_min
            best_key  = f'{_NOTE_NAMES[root]} Minor'

    # Normalise correlation [-1,1] → confidence [0,1]
    confidence = float((best_corr + 1.0) / 2.0)
    return best_key, confidence


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

def compute_mel_patches_discogs(y16: np.ndarray) -> np.ndarray:
    """Compute log-mel spectrogram patches matching Essentia's EffNet input spec.

    Spec: n_fft=512, hop=256, n_mels=96, fmin=0, fmax=8000, power=2
          → unit-sum normalise each frame → natural log compress
          → overlapping [128-frame × 96-mel] patches with 50% overlap

    Returns float32 array of shape (N_patches, PATCH_FRAMES, N_MELS).
    """
    mel = librosa.feature.melspectrogram(
        y=y16, sr=SR_DISCOGS,
        n_fft=N_FFT_D, hop_length=HOP_LEN_D,
        n_mels=N_MELS_D, fmin=FMIN_D, fmax=FMAX_D,
        power=2.0,
    )   # (96, T)
    col_sums = mel.sum(axis=0, keepdims=True)
    mel = mel / np.maximum(col_sums, 1e-10)   # unit-sum normalise each frame
    mel = np.log(mel + 1e-9)                  # natural log compression
    mel = mel.T.astype(np.float32)            # (T, 96)
    patches = []
    for start in range(0, len(mel) - PATCH_FRAMES + 1, PATCH_HOP):
        patches.append(mel[start:start + PATCH_FRAMES])   # (128, 96)
    return np.array(patches, dtype=np.float32) if patches else None   # (N, 128, 96)


def run_discogs_inference(patches: np.ndarray) -> np.ndarray:
    """Run Discogs-EffNet on patches, return embeddings (N_patches, 1280).

    Pads last batch to BATCH_SIZE_D=64 (required by the frozen graph).
    """
    all_emb = []
    for i in range(0, len(patches), BATCH_SIZE_D):
        batch  = patches[i:i + BATCH_SIZE_D]
        actual = len(batch)
        if actual < BATCH_SIZE_D:
            pad   = np.zeros((BATCH_SIZE_D - actual, PATCH_FRAMES, N_MELS_D), dtype=np.float32)
            batch = np.concatenate([batch, pad], axis=0)
        emb = discogs_session.run(discogs_embed_tensor, {discogs_input_tensor: batch})  # (64, 1280)
        all_emb.append(emb[:actual])   # drop padding rows
    return np.concatenate(all_emb, axis=0)   # (N_patches, 1280)


# 🔥 THIS IS THE FIX: Load models AFTER the server port opens
@app.on_event("startup")
async def load_all_models():
    global yamnet_model, mood_model, mood_encoder, feature_scaler
    global instrument_model, instrument_encoder
    global stage1_model, stage1_encoder, stage2_models, stage2_encoders
    global discogs_session, discogs_input_tensor, discogs_embed_tensor
    print("🚪 Port is open! Now loading AI brains in the background...")

    # ── YAMNet (still used for mood prediction) ──
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

    # ── Discogs-EfficientNet frozen graph (used for genre feature extraction) ──
    discogs_pb = os.path.join(BASE_DIR, 'discogs-effnet-bs64-1.pb')
    discogs_graph = tf.Graph()
    with discogs_graph.as_default():
        graph_def = tf.compat.v1.GraphDef()
        with open(discogs_pb, 'rb') as fh:
            graph_def.ParseFromString(fh.read())
        tf.import_graph_def(graph_def, name='')
    discogs_session      = tf.compat.v1.Session(graph=discogs_graph)
    discogs_input_tensor = discogs_graph.get_tensor_by_name('serving_default_melspectrogram:0')
    discogs_embed_tensor = discogs_graph.get_tensor_by_name('PartitionedCall:1')
    print(f"✅ Discogs-EfficientNet loaded from {discogs_pb}")

    # ── Feature scaler (required — normalises 2641-dim Discogs features) ──
    with open(os.path.join(BASE_DIR, 'feature_scaler.pkl'), 'rb') as f:
        feature_scaler = pickle.load(f)

    # ── Mood model (trained on 1024-dim YAMNet mean embeddings) ──
    mood_model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, 'yamnet_mood_model.h5'),
        custom_objects={'loss_fn': focal_loss(gamma=2.0, alpha=0.25)},
        compile=False,
    )
    with open(os.path.join(BASE_DIR, 'yamnet_mood_model_encoder.pkl'), 'rb') as f:
        mood_encoder = pickle.load(f)

    # ── Instrument model (trained on 1024-dim YAMNet embeddings) ──
    inst_model_path = os.path.join(BASE_DIR, 'instrument_model.h5')
    if os.path.exists(inst_model_path):
        instrument_model = tf.keras.models.load_model(
            inst_model_path,
            custom_objects={'loss_fn': focal_loss(gamma=2.0, alpha=0.25)},
            compile=False,
        )
        with open(os.path.join(BASE_DIR, 'instrument_model_encoder.pkl'), 'rb') as f:
            instrument_encoder = pickle.load(f)
        print("✅ Instrument model loaded")

    # ── Hierarchical genre models (trained on 2641-dim Discogs features) ──
    stage1_model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, 'stage1_model.h5'),
        compile=False,
    )
    with open(os.path.join(BASE_DIR, 'stage1_model_encoder.pkl'), 'rb') as f:
        stage1_encoder = pickle.load(f)

    for cat in STAGE2_CATEGORIES:
        model_path   = os.path.join(BASE_DIR, f'stage2_{cat}_model.h5')
        encoder_path = os.path.join(BASE_DIR, f'stage2_{cat}_model_encoder.pkl')
        if os.path.exists(model_path):
            stage2_models[cat] = tf.keras.models.load_model(
                model_path,
                custom_objects={'loss_fn': focal_loss(gamma=2.0, alpha=0.25)},
                compile=False,
            )
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                stage2_encoders[cat] = pickle.load(f)

    loaded_s2 = sum(1 for cat in STAGE2_CATEGORIES if cat in stage2_models)
    print(f"✅ Feature scaler loaded  (2641-dim normalisation)")
    print(f"✅ Genre models: Stage 1 + {loaded_s2}/{len(STAGE2_CATEGORIES)} Stage 2 models loaded")
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
    if yamnet_model is None or stage1_model is None or discogs_session is None:
        raise HTTPException(status_code=503, detail="AI is still warming up. Try again in 30 seconds!")
        
    validate_audio_upload(file)
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE_MB}MB.")
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(contents)

    try:
        # ── Load at 22050 Hz for librosa features ─────────────────────────────
        DURATION = 30
        y22, _ = librosa.load(temp_path, sr=SR_LIBROSA, duration=DURATION, mono=True)
        target_22k = DURATION * SR_LIBROSA
        if len(y22) < target_22k:
            y22 = np.pad(y22, (0, target_22k - len(y22)))
        else:
            y22 = y22[:target_22k]
        peak = np.abs(y22).max()
        if peak > 0:
            y22 = y22 / peak

        # ── Librosa features (must match extract_discogs_features.py exactly) ──
        mfcc      = librosa.feature.mfcc(y=y22, sr=SR_LIBROSA, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        chroma    = librosa.feature.chroma_stft(y=y22, sr=SR_LIBROSA, hop_length=HOP_LENGTH)
        contrast  = librosa.feature.spectral_contrast(y=y22, sr=SR_LIBROSA, hop_length=HOP_LENGTH)
        rms       = librosa.feature.rms(y=y22, hop_length=HOP_LENGTH)[0]
        onset_env = librosa.onset.onset_strength(y=y22, sr=SR_LIBROSA, hop_length=HOP_LENGTH)
        raw_tempo = float(librosa.beat.tempo(onset_envelope=onset_env, sr=SR_LIBROSA, hop_length=HOP_LENGTH)[0])

        # ── Musical key detection — uses chromagram already computed above ──────
        detected_key, key_confidence = detect_key(chroma)

        # ── LUFS loudness — ITU-R BS.1770 integrated loudness ────────────────
        lufs, lufs_note = compute_lufs(y22, SR_LIBROSA)

        # ── Time signature ────────────────────────────────────────────────────
        time_signature = detect_time_signature(y22, SR_LIBROSA, HOP_LENGTH)

        librosa_feats = np.concatenate([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),         # 40
            np.mean(chroma, axis=1), np.std(chroma, axis=1),     # 24
            np.mean(contrast, axis=1), np.std(contrast, axis=1), # 14
            [np.mean(rms)], [np.std(rms)],                       #  2
            [np.clip(raw_tempo / 240.0, 0.0, 1.0)],              #  1
        ])  # 81 features

        # ── Resample to 16 kHz (shared for YAMNet mood + Discogs genre) ──────
        y16 = librosa.resample(y22, orig_sr=SR_LIBROSA, target_sr=SR_DISCOGS)
        target_16k = DURATION * SR_DISCOGS
        if len(y16) < target_16k:
            y16 = np.pad(y16, (0, target_16k - len(y16)))
        else:
            y16 = y16[:target_16k]
        y16 = y16.astype(np.float32)
        peak16 = np.abs(y16).max()
        if peak16 > 0:
            y16 = y16 / peak16

        # ── YAMNet — class scores for vocal detection + embeddings for mood ──────
        class_scores, embeddings, _ = yamnet_model(y16)
        class_scores_np = class_scores.numpy()              # (N_frames, 521)
        emb_np          = embeddings.numpy()
        yamnet_mean     = np.mean(emb_np, axis=0)           # (1024,)

        # Vocal detection uses raw class scores before any pooling
        vocals, vocals_confidence = detect_vocals(class_scores_np)

        # ── Discogs-EffNet embeddings — used for genre prediction (2560-dim) ──
        patches = compute_mel_patches_discogs(y16)
        if patches is None or len(patches) == 0:
            raise ValueError("Audio too short to generate Discogs mel patches")
        discogs_embs = run_discogs_inference(patches)         # (N_patches, 1280)
        discogs_mean = np.mean(discogs_embs, axis=0)          # (1280,)
        discogs_std  = np.std(discogs_embs, axis=0)           # (1280,)

        # ── Combine → 2641-dim → scale ────────────────────────────────────────
        raw_feat = np.concatenate([discogs_mean, discogs_std, librosa_feats])  # (2641,)
        X_genre  = feature_scaler.transform(raw_feat[np.newaxis, :])           # (1, 2641)

        # ── Mood + instrument both use 1024-dim YAMNet mean embedding ───────────
        X_mood = yamnet_mean[np.newaxis, :]   # (1, 1024)
        mood_preds = mood_model.predict(X_mood, verbose=0)

        # ── Instrument prediction ─────────────────────────────────────────────
        detected_instruments = []
        if instrument_model is not None and instrument_encoder is not None:
            inst_preds  = instrument_model.predict(X_mood, verbose=0)[0]
            # Return top instruments above confidence threshold, max 3
            INST_THRESHOLD = 0.35
            top_inst_idx = np.argsort(inst_preds)[::-1]
            for idx in top_inst_idx:
                conf = float(inst_preds[idx])
                if conf < INST_THRESHOLD or len(detected_instruments) >= 3:
                    break
                name = instrument_encoder.classes_[idx]
                detected_instruments.append({"instrument": name, "confidence": round(conf, 3)})

        # ── Genre prediction: Stage 1 as soft filter + Stage 2 for specific genre
        #
        # Stage 1 predicts which broad category the track belongs to.
        # Even at ~48% hard accuracy, its probability distribution is useful:
        # a track that Stage 1 gives 5% Latin probability should not be classified
        # as Bossa Nova just because Stage2_Latin is overconfident.
        #
        # Scoring formula:
        #   stage1_weight  = Stage 1 softmax probability for this category
        #   stage2_adj     = (stage2_raw - 1/n) / (1 - 1/n)  [normalised: 0=random, 1=certain]
        #   final_score    = stage2_adj * (stage1_weight ^ STAGE1_INFLUENCE)
        # The normalised formula is critical: models with more classes had a scoring
        # advantage under the old "raw - 1/n" formula (e.g. Latin with 12 genres only
        # deducted 0.083 vs Ambient with 4 genres deducting 0.25 — same raw softmax
        # gave Latin a 2.8x scoring edge). Normalising puts all Stage 2 models on an
        # equal footing regardless of how many sub-genres they contain.
        #
        # STAGE1_INFLUENCE controls how hard Stage 1 filters Stage 2:
        #   0.0 = ignore Stage 1 entirely (Stage 2 competes purely on its own)
        #   0.2 = very light touch (Stage 1 only breaks near-ties)
        #   0.5 = soft filter (Stage 1 nudges but doesn't dominate)
        #   1.0 = full multiplication (Stage 1 probability directly scales Stage 2)
        # Using 0.2: Stage 1 is only ~45% accurate so we let Stage 2 lead.
        # The well-trained Stage 2 models (Ambient 69%, Folk 68%, Classical 67%)
        # are far more reliable than Stage 1's category routing.

        STAGE1_INFLUENCE = 0.2

        # Run Stage 1 — get per-category probability distribution
        stage1_preds = stage1_model.predict(X_genre, verbose=0)[0]  # shape: (n_categories,)
        stage1_cat_probs = {
            str(stage1_encoder.inverse_transform([i])[0]): float(stage1_preds[i])
            for i in range(len(stage1_encoder.classes_))
        }
        print(f"Stage 1 category probs: { {k: round(v,3) for k,v in sorted(stage1_cat_probs.items(), key=lambda x: -x[1])} }")

        all_candidates = []  # (final_score, raw_conf, genre_label)

        for _cat in STAGE2_CATEGORIES:
            _s2_model   = stage2_models.get(_cat)
            _s2_encoder = stage2_encoders.get(_cat)
            if _s2_encoder is None:
                continue
            n_cls = len(_s2_encoder.classes_)
            _s1_weight = stage1_cat_probs.get(_cat, 1.0 / len(STAGE2_CATEGORIES))

            if _s2_model is not None:
                _preds = _s2_model.predict(X_genre, verbose=0)
                for _i in range(n_cls):
                    _genre  = str(_s2_encoder.inverse_transform([_i])[0])
                    _raw    = float(_preds[0][_i])
                    _adj    = (_raw - (1.0 / n_cls)) / (1.0 - (1.0 / n_cls))
                    _final  = _adj * (_s1_weight ** STAGE1_INFLUENCE)
                    all_candidates.append((_final, _raw, _genre))
            else:
                # Single-genre category — add with neutral score weighted by Stage 1
                _genre  = str(_s2_encoder.classes_[0])
                _final  = 0.0
                all_candidates.append((_final, 1.0 / n_cls, _genre))

        # Sort by final score descending
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        print(f"Top 5 genre candidates: {[(g, round(s,4)) for s,_,g in all_candidates[:5]]}")

        primary_genre        = all_candidates[0][2]
        primary_genre_conf   = all_candidates[0][1]
        secondary_genre      = all_candidates[1][2] if len(all_candidates) > 1 else primary_genre
        secondary_genre_conf = all_candidates[1][1] if len(all_candidates) > 1 else primary_genre_conf
        tertiary_genre       = all_candidates[2][2] if len(all_candidates) > 2 else secondary_genre
        tertiary_genre_conf  = all_candidates[2][1] if len(all_candidates) > 2 else secondary_genre_conf

        # ── Top 3 moods — model knows 14, return ranked top 3 ────────────────
        mood_scores = mood_preds[0]
        top_mood_indices = np.argsort(mood_scores)[::-1][:3]
        top_moods = [
            {
                "mood":       str(mood_encoder.inverse_transform([i])[0]),
                "confidence": float(mood_scores[i]),
            }
            for i in top_mood_indices
        ]
        mood_result = top_moods[0]["mood"]
        mood_conf   = top_moods[0]["confidence"]

        # ── BPM — round to nearest integer ───────────────────────────────────
        bpm = int(round(raw_tempo))

        # ── Energy level — derived from RMS, mapped to 1–10 scale ────────────
        # RMS is the root-mean-square amplitude of the audio signal.
        # Typical music RMS range: ~0.01 (very quiet/ambient) to ~0.25 (loud/energetic).
        # We log-scale it into a 1–10 integer so it matches industry standard metadata.
        rms_mean = float(np.mean(rms))
        rms_clipped = float(np.clip(rms_mean, 0.005, 0.25))
        import math
        energy_raw = (math.log(rms_clipped) - math.log(0.005)) / (math.log(0.25) - math.log(0.005))
        energy_level = int(round(1 + energy_raw * 9))   # 1 = very calm, 10 = very energetic

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
            "bossa_nova":        "Bossa Nova",
            "caribbean_dance":   "Bachata",   # Bachata+Merengue merge → show Bachata (more recognisable)
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
            # Merged class labels from retrained Stage 2 models
            "classical_baroque": "Classical",   # Classical + Baroque merged
            "folk_country":      "Folk",        # Folk + Country merged
            "pop_acoustic":      "Pop",         # Pop + Acoustic merged
            "jazz_soul":         "Jazz",        # Jazz + Blues + Funk/Soul merged
            "urban":             "Hip-Hop",     # Hip-Hop + Trap + R&B + Afrobeats merged
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

        clean_genre     = normalize_genre(primary_genre)
        clean_secondary = normalize_genre(secondary_genre)
        clean_tertiary  = normalize_genre(tertiary_genre)
        clean_mood      = normalize_mood(mood_result)
        clean_moods     = [
            {"mood": normalize_mood(m["mood"]), "confidence": round(m["confidence"], 3)}
            for m in top_moods
        ]

        tempo_descriptor = get_tempo_descriptor(bpm)
        use_cases        = get_use_cases(clean_genre, clean_moods, energy_level, vocals)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            # ── Genre (primary + secondary + tertiary) ──
            "genre":                      clean_genre,
            "genre_confidence":           round(primary_genre_conf, 3),
            "secondary_genre":            clean_secondary,
            "secondaryGenre":             clean_secondary,      # camelCase legacy
            "secondary_genre_confidence": round(secondary_genre_conf, 3),
            "tertiary_genre":             clean_tertiary,
            "tertiary_genre_confidence":  round(tertiary_genre_conf, 3),
            # ── Mood (primary + top 3 ranked list) ──
            "mood":                       clean_mood,
            "mood_confidence":            round(mood_conf, 3),
            "moods":                      clean_moods,
            # ── Musical key ──
            "key":                        detected_key,
            "key_confidence":             round(key_confidence, 3),
            # ── Tempo & energy ──
            "bpm":                        bpm,
            "tempo":                      tempo_descriptor,     # "Slow" / "Moderate" / "Upbeat" / "Fast"
            "energy":                     energy_level,         # 1 (calm) → 10 (intense)
            # ── Instrument detection ──
            "instruments":                detected_instruments,  # e.g. [{"instrument":"Piano","confidence":0.72}]
            # ── Vocal detection ──
            "vocals":                     vocals,
            "vocals_confidence":          vocals_confidence,
            # ── Loudness ──
            "loudness_lufs":              lufs,
            "loudness_note":              lufs_note,
            # ── Time signature ──
            "time_signature":             time_signature,
            # ── Use case / placement tags ──
            "use_cases":                  use_cases,
            # ── Legacy ──
            "predictions":                [clean_genre, clean_mood],
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
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)