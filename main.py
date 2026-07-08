from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import tensorflow as tf
import librosa
import numpy as np
import pickle
import os
import json
import subprocess
import concurrent.futures
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional

load_dotenv()

# Flag set to True only after all AI models finish loading.
# Used by /health so Cloud Run startup probe knows when the instance is ready.
_models_ready = False

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

# ── MTG genre_discogs400 — 400 Discogs genre labels in model output order ─────
DISCOGS400_CLASSES = [
    "Blues---Boogie Woogie","Blues---Chicago Blues","Blues---Country Blues",
    "Blues---Delta Blues","Blues---Electric Blues","Blues---Harmonica Blues",
    "Blues---Jump Blues","Blues---Louisiana Blues","Blues---Modern Electric Blues",
    "Blues---Piano Blues","Blues---Rhythm & Blues","Blues---Texas Blues",
    "Brass & Military---Brass Band","Brass & Military---Marches","Brass & Military---Military",
    "Children's---Educational","Children's---Nursery Rhymes","Children's---Story",
    "Classical---Baroque","Classical---Choral","Classical---Classical",
    "Classical---Contemporary","Classical---Impressionist","Classical---Medieval",
    "Classical---Modern","Classical---Neo-Classical","Classical---Neo-Romantic",
    "Classical---Opera","Classical---Post-Modern","Classical---Renaissance",
    "Classical---Romantic","Electronic---Abstract","Electronic---Acid",
    "Electronic---Acid House","Electronic---Acid Jazz","Electronic---Ambient",
    "Electronic---Bassline","Electronic---Beatdown","Electronic---Berlin-School",
    "Electronic---Big Beat","Electronic---Bleep","Electronic---Breakbeat",
    "Electronic---Breakcore","Electronic---Breaks","Electronic---Broken Beat",
    "Electronic---Chillwave","Electronic---Chiptune","Electronic---Dance-pop",
    "Electronic---Dark Ambient","Electronic---Darkwave","Electronic---Deep House",
    "Electronic---Deep Techno","Electronic---Disco","Electronic---Disco Polo",
    "Electronic---Donk","Electronic---Downtempo","Electronic---Drone",
    "Electronic---Drum n Bass","Electronic---Dub","Electronic---Dub Techno",
    "Electronic---Dubstep","Electronic---Dungeon Synth","Electronic---EBM",
    "Electronic---Electro","Electronic---Electro House","Electronic---Electroclash",
    "Electronic---Euro House","Electronic---Euro-Disco","Electronic---Eurobeat",
    "Electronic---Eurodance","Electronic---Experimental","Electronic---Freestyle",
    "Electronic---Future Jazz","Electronic---Gabber","Electronic---Garage House",
    "Electronic---Ghetto","Electronic---Ghetto House","Electronic---Glitch",
    "Electronic---Goa Trance","Electronic---Grime","Electronic---Halftime",
    "Electronic---Hands Up","Electronic---Happy Hardcore","Electronic---Hard House",
    "Electronic---Hard Techno","Electronic---Hard Trance","Electronic---Hardcore",
    "Electronic---Hardstyle","Electronic---Hi NRG","Electronic---Hip Hop",
    "Electronic---Hip-House","Electronic---House","Electronic---IDM",
    "Electronic---Illbient","Electronic---Industrial","Electronic---Italo House",
    "Electronic---Italo-Disco","Electronic---Italodance","Electronic---Jazzdance",
    "Electronic---Juke","Electronic---Jumpstyle","Electronic---Jungle",
    "Electronic---Latin","Electronic---Leftfield","Electronic---Makina",
    "Electronic---Minimal","Electronic---Minimal Techno","Electronic---Modern Classical",
    "Electronic---Musique Concrète","Electronic---Neofolk","Electronic---New Age",
    "Electronic---New Beat","Electronic---New Wave","Electronic---Noise",
    "Electronic---Nu-Disco","Electronic---Power Electronics","Electronic---Progressive Breaks",
    "Electronic---Progressive House","Electronic---Progressive Trance","Electronic---Psy-Trance",
    "Electronic---Rhythmic Noise","Electronic---Schranz","Electronic---Sound Collage",
    "Electronic---Speed Garage","Electronic---Speedcore","Electronic---Synth-pop",
    "Electronic---Synthwave","Electronic---Tech House","Electronic---Tech Trance",
    "Electronic---Techno","Electronic---Trance","Electronic---Tribal",
    "Electronic---Tribal House","Electronic---Trip Hop","Electronic---Tropical House",
    "Electronic---UK Garage","Electronic---Vaporwave",
    "Folk, World, & Country---African","Folk, World, & Country---Bluegrass",
    "Folk, World, & Country---Cajun","Folk, World, & Country---Canzone Napoletana",
    "Folk, World, & Country---Catalan Music","Folk, World, & Country---Celtic",
    "Folk, World, & Country---Country","Folk, World, & Country---Fado",
    "Folk, World, & Country---Flamenco","Folk, World, & Country---Folk",
    "Folk, World, & Country---Gospel","Folk, World, & Country---Highlife",
    "Folk, World, & Country---Hillbilly","Folk, World, & Country---Hindustani",
    "Folk, World, & Country---Honky Tonk","Folk, World, & Country---Indian Classical",
    "Folk, World, & Country---Laïkó","Folk, World, & Country---Nordic",
    "Folk, World, & Country---Pacific","Folk, World, & Country---Polka",
    "Folk, World, & Country---Raï","Folk, World, & Country---Romani",
    "Folk, World, & Country---Soukous","Folk, World, & Country---Séga",
    "Folk, World, & Country---Volksmusik","Folk, World, & Country---Zouk",
    "Folk, World, & Country---Éntekhno",
    "Funk / Soul---Afrobeat","Funk / Soul---Boogie","Funk / Soul---Contemporary R&B",
    "Funk / Soul---Disco","Funk / Soul---Free Funk","Funk / Soul---Funk",
    "Funk / Soul---Gospel","Funk / Soul---Neo Soul","Funk / Soul---New Jack Swing",
    "Funk / Soul---P.Funk","Funk / Soul---Psychedelic","Funk / Soul---Rhythm & Blues",
    "Funk / Soul---Soul","Funk / Soul---Swingbeat","Funk / Soul---UK Street Soul",
    "Hip Hop---Bass Music","Hip Hop---Boom Bap","Hip Hop---Bounce",
    "Hip Hop---Britcore","Hip Hop---Cloud Rap","Hip Hop---Conscious",
    "Hip Hop---Crunk","Hip Hop---Cut-up/DJ","Hip Hop---DJ Battle Tool",
    "Hip Hop---Electro","Hip Hop---G-Funk","Hip Hop---Gangsta",
    "Hip Hop---Grime","Hip Hop---Hardcore Hip-Hop","Hip Hop---Horrorcore",
    "Hip Hop---Instrumental","Hip Hop---Jazzy Hip-Hop","Hip Hop---Miami Bass",
    "Hip Hop---Pop Rap","Hip Hop---Ragga HipHop","Hip Hop---RnB/Swing",
    "Hip Hop---Screw","Hip Hop---Thug Rap","Hip Hop---Trap",
    "Hip Hop---Trip Hop","Hip Hop---Turntablism",
    "Jazz---Afro-Cuban Jazz","Jazz---Afrobeat","Jazz---Avant-garde Jazz",
    "Jazz---Big Band","Jazz---Bop","Jazz---Bossa Nova",
    "Jazz---Contemporary Jazz","Jazz---Cool Jazz","Jazz---Dixieland",
    "Jazz---Easy Listening","Jazz---Free Improvisation","Jazz---Free Jazz",
    "Jazz---Fusion","Jazz---Gypsy Jazz","Jazz---Hard Bop",
    "Jazz---Jazz-Funk","Jazz---Jazz-Rock","Jazz---Latin Jazz",
    "Jazz---Modal","Jazz---Post Bop","Jazz---Ragtime",
    "Jazz---Smooth Jazz","Jazz---Soul-Jazz","Jazz---Space-Age","Jazz---Swing",
    "Latin---Afro-Cuban","Latin---Baião","Latin---Batucada",
    "Latin---Beguine","Latin---Bolero","Latin---Boogaloo",
    "Latin---Bossanova","Latin---Cha-Cha","Latin---Charanga",
    "Latin---Compas","Latin---Cubano","Latin---Cumbia",
    "Latin---Descarga","Latin---Forró","Latin---Guaguancó",
    "Latin---Guajira","Latin---Guaracha","Latin---MPB",
    "Latin---Mambo","Latin---Mariachi","Latin---Merengue",
    "Latin---Norteño","Latin---Nueva Cancion","Latin---Pachanga",
    "Latin---Porro","Latin---Ranchera","Latin---Reggaeton",
    "Latin---Rumba","Latin---Salsa","Latin---Samba",
    "Latin---Son","Latin---Son Montuno","Latin---Tango",
    "Latin---Tejano","Latin---Vallenato",
    "Non-Music---Audiobook","Non-Music---Comedy","Non-Music---Dialogue",
    "Non-Music---Education","Non-Music---Field Recording","Non-Music---Interview",
    "Non-Music---Monolog","Non-Music---Poetry","Non-Music---Political",
    "Non-Music---Promotional","Non-Music---Radioplay","Non-Music---Religious",
    "Non-Music---Spoken Word",
    "Pop---Ballad","Pop---Bollywood","Pop---Bubblegum",
    "Pop---Chanson","Pop---City Pop","Pop---Europop",
    "Pop---Indie Pop","Pop---J-pop","Pop---K-pop",
    "Pop---Kayōkyoku","Pop---Light Music","Pop---Music Hall",
    "Pop---Novelty","Pop---Parody","Pop---Schlager","Pop---Vocal",
    "Reggae---Calypso","Reggae---Dancehall","Reggae---Dub",
    "Reggae---Lovers Rock","Reggae---Ragga","Reggae---Reggae",
    "Reggae---Reggae-Pop","Reggae---Rocksteady","Reggae---Roots Reggae",
    "Reggae---Ska","Reggae---Soca",
    "Rock---AOR","Rock---Acid Rock","Rock---Acoustic",
    "Rock---Alternative Rock","Rock---Arena Rock","Rock---Art Rock",
    "Rock---Atmospheric Black Metal","Rock---Avantgarde","Rock---Beat",
    "Rock---Black Metal","Rock---Blues Rock","Rock---Brit Pop",
    "Rock---Classic Rock","Rock---Coldwave","Rock---Country Rock",
    "Rock---Crust","Rock---Death Metal","Rock---Deathcore",
    "Rock---Deathrock","Rock---Depressive Black Metal","Rock---Doo Wop",
    "Rock---Doom Metal","Rock---Dream Pop","Rock---Emo",
    "Rock---Ethereal","Rock---Experimental","Rock---Folk Metal",
    "Rock---Folk Rock","Rock---Funeral Doom Metal","Rock---Funk Metal",
    "Rock---Garage Rock","Rock---Glam","Rock---Goregrind",
    "Rock---Goth Rock","Rock---Gothic Metal","Rock---Grindcore",
    "Rock---Grunge","Rock---Hard Rock","Rock---Hardcore",
    "Rock---Heavy Metal","Rock---Indie Rock","Rock---Industrial",
    "Rock---Krautrock","Rock---Lo-Fi","Rock---Lounge",
    "Rock---Math Rock","Rock---Melodic Death Metal","Rock---Melodic Hardcore",
    "Rock---Metalcore","Rock---Mod","Rock---Neofolk",
    "Rock---New Wave","Rock---No Wave","Rock---Noise",
    "Rock---Noisecore","Rock---Nu Metal","Rock---Oi",
    "Rock---Parody","Rock---Pop Punk","Rock---Pop Rock",
    "Rock---Pornogrind","Rock---Post Rock","Rock---Post-Hardcore",
    "Rock---Post-Metal","Rock---Post-Punk","Rock---Power Metal",
    "Rock---Power Pop","Rock---Power Violence","Rock---Prog Rock",
    "Rock---Progressive Metal","Rock---Psychedelic Rock","Rock---Psychobilly",
    "Rock---Pub Rock","Rock---Punk","Rock---Rock & Roll",
    "Rock---Rockabilly","Rock---Shoegaze","Rock---Ska",
    "Rock---Sludge Metal","Rock---Soft Rock","Rock---Southern Rock",
    "Rock---Space Rock","Rock---Speed Metal","Rock---Stoner Rock",
    "Rock---Surf","Rock---Symphonic Rock","Rock---Technical Death Metal",
    "Rock---Thrash","Rock---Twist","Rock---Viking Metal","Rock---Yé-Yé",
    "Stage & Screen---Musical","Stage & Screen---Score",
    "Stage & Screen---Soundtrack","Stage & Screen---Theme",
]

# Maps each of the 400 Discogs labels → one of the ALLOWED_GENRES strings.
# None means skip (Non-Music, Brass & Military with no clean mapping).
DISCOGS400_TO_GENRE = {
    # Blues
    "Blues---Boogie Woogie": "Blues", "Blues---Chicago Blues": "Blues",
    "Blues---Country Blues": "Blues", "Blues---Delta Blues": "Blues",
    "Blues---Electric Blues": "Blues", "Blues---Harmonica Blues": "Blues",
    "Blues---Jump Blues": "Blues", "Blues---Louisiana Blues": "Blues",
    "Blues---Modern Electric Blues": "Blues", "Blues---Piano Blues": "Blues",
    "Blues---Rhythm & Blues": "R&B", "Blues---Texas Blues": "Blues",
    # Brass & Military — no clean genre match
    "Brass & Military---Brass Band": None, "Brass & Military---Marches": None,
    "Brass & Military---Military": None,
    # Children's
    "Children's---Educational": "Children's", "Children's---Nursery Rhymes": "Children's",
    "Children's---Story": "Children's",
    # Classical
    "Classical---Baroque": "Baroque", "Classical---Choral": "Classical",
    "Classical---Classical": "Classical", "Classical---Contemporary": "Classical",
    "Classical---Impressionist": "Classical", "Classical---Medieval": "Classical",
    "Classical---Modern": "Classical", "Classical---Neo-Classical": "Classical",
    "Classical---Neo-Romantic": "Classical", "Classical---Opera": "Opera",
    "Classical---Post-Modern": "Classical", "Classical---Renaissance": "Classical",
    "Classical---Romantic": "Classical",
    # Electronic
    "Electronic---Abstract": "Electronic", "Electronic---Acid": "Electronic",
    "Electronic---Acid House": "House", "Electronic---Acid Jazz": "Jazz",
    "Electronic---Ambient": "Ambient", "Electronic---Bassline": "Electronic",
    "Electronic---Beatdown": "Electronic", "Electronic---Berlin-School": "Electronic",
    "Electronic---Big Beat": "Electronic", "Electronic---Bleep": "Electronic",
    "Electronic---Breakbeat": "Electronic", "Electronic---Breakcore": "Electronic",
    "Electronic---Breaks": "Electronic", "Electronic---Broken Beat": "Electronic",
    "Electronic---Chillwave": "Ambient", "Electronic---Chiptune": "Electronic",
    "Electronic---Dance-pop": "Pop", "Electronic---Dark Ambient": "Ambient",
    "Electronic---Darkwave": "Electronic", "Electronic---Deep House": "House",
    "Electronic---Deep Techno": "Techno", "Electronic---Disco": "Electronic",
    "Electronic---Disco Polo": "Electronic", "Electronic---Donk": "Electronic",
    "Electronic---Downtempo": "Ambient", "Electronic---Drone": "Ambient",
    "Electronic---Drum n Bass": "Drum & Bass", "Electronic---Dub": "Reggae",
    "Electronic---Dub Techno": "Techno", "Electronic---Dubstep": "Dubstep",
    "Electronic---Dungeon Synth": "Electronic", "Electronic---EBM": "Electronic",
    "Electronic---Electro": "Electronic", "Electronic---Electro House": "House",
    "Electronic---Electroclash": "Electronic", "Electronic---Euro House": "House",
    "Electronic---Euro-Disco": "Electronic", "Electronic---Eurobeat": "Electronic",
    "Electronic---Eurodance": "Electronic", "Electronic---Experimental": "Electronic",
    "Electronic---Freestyle": "Electronic", "Electronic---Future Jazz": "Jazz",
    "Electronic---Gabber": "Electronic", "Electronic---Garage House": "House",
    "Electronic---Ghetto": "Electronic", "Electronic---Ghetto House": "House",
    "Electronic---Glitch": "Electronic", "Electronic---Goa Trance": "Trance",
    "Electronic---Grime": "Hip-Hop", "Electronic---Halftime": "Electronic",
    "Electronic---Hands Up": "Electronic", "Electronic---Happy Hardcore": "Electronic",
    "Electronic---Hard House": "House", "Electronic---Hard Techno": "Techno",
    "Electronic---Hard Trance": "Trance", "Electronic---Hardcore": "Electronic",
    "Electronic---Hardstyle": "Electronic", "Electronic---Hi NRG": "Electronic",
    "Electronic---Hip Hop": "Hip-Hop", "Electronic---Hip-House": "House",
    "Electronic---House": "House", "Electronic---IDM": "Electronic",
    "Electronic---Illbient": "Electronic", "Electronic---Industrial": "Electronic",
    "Electronic---Italo House": "House", "Electronic---Italo-Disco": "Electronic",
    "Electronic---Italodance": "Electronic", "Electronic---Jazzdance": "Electronic",
    "Electronic---Juke": "Electronic", "Electronic---Jumpstyle": "Electronic",
    "Electronic---Jungle": "Drum & Bass", "Electronic---Latin": "Latin",
    "Electronic---Leftfield": "Electronic", "Electronic---Makina": "Electronic",
    "Electronic---Minimal": "Electronic", "Electronic---Minimal Techno": "Techno",
    "Electronic---Modern Classical": "Classical", "Electronic---Musique Concrète": "Electronic",
    "Electronic---Neofolk": "Folk", "Electronic---New Age": "New Age",
    "Electronic---New Beat": "Electronic", "Electronic---New Wave": "Electronic",
    "Electronic---Noise": "Electronic", "Electronic---Nu-Disco": "Electronic",
    "Electronic---Power Electronics": "Electronic", "Electronic---Progressive Breaks": "Electronic",
    "Electronic---Progressive House": "House", "Electronic---Progressive Trance": "Trance",
    "Electronic---Psy-Trance": "Trance", "Electronic---Rhythmic Noise": "Electronic",
    "Electronic---Schranz": "Techno", "Electronic---Sound Collage": "Electronic",
    "Electronic---Speed Garage": "Electronic", "Electronic---Speedcore": "Electronic",
    "Electronic---Synth-pop": "Electronic", "Electronic---Synthwave": "Synthwave",
    "Electronic---Tech House": "House", "Electronic---Tech Trance": "Trance",
    "Electronic---Techno": "Techno", "Electronic---Trance": "Trance",
    "Electronic---Tribal": "Electronic", "Electronic---Tribal House": "House",
    "Electronic---Trip Hop": "Ambient", "Electronic---Tropical House": "House",
    "Electronic---UK Garage": "Electronic", "Electronic---Vaporwave": "Electronic",
    # Folk, World & Country
    "Folk, World, & Country---African": "Afrobeats",
    "Folk, World, & Country---Bluegrass": "Folk",
    "Folk, World, & Country---Cajun": "Folk",
    "Folk, World, & Country---Canzone Napoletana": "World Music",
    "Folk, World, & Country---Catalan Music": "World Music",
    "Folk, World, & Country---Celtic": "Folk",
    "Folk, World, & Country---Country": "Country",
    "Folk, World, & Country---Fado": "World Music",
    "Folk, World, & Country---Flamenco": "Flamenco",
    "Folk, World, & Country---Folk": "Folk",
    "Folk, World, & Country---Gospel": "Gospel",
    "Folk, World, & Country---Highlife": "Afrobeats",
    "Folk, World, & Country---Hillbilly": "Country",
    "Folk, World, & Country---Hindustani": "World Music",
    "Folk, World, & Country---Honky Tonk": "Country",
    "Folk, World, & Country---Indian Classical": "World Music",
    "Folk, World, & Country---Laïkó": "World Music",
    "Folk, World, & Country---Nordic": "World Music",
    "Folk, World, & Country---Pacific": "World Music",
    "Folk, World, & Country---Polka": "World Music",
    "Folk, World, & Country---Raï": "World Music",
    "Folk, World, & Country---Romani": "World Music",
    "Folk, World, & Country---Soukous": "Afrobeats",
    "Folk, World, & Country---Séga": "World Music",
    "Folk, World, & Country---Volksmusik": "World Music",
    "Folk, World, & Country---Zouk": "Latin",
    "Folk, World, & Country---Éntekhno": "World Music",
    # Funk / Soul
    "Funk / Soul---Afrobeat": "Afrobeats", "Funk / Soul---Boogie": "Funk/Soul",
    "Funk / Soul---Contemporary R&B": "R&B", "Funk / Soul---Disco": "Funk/Soul",
    "Funk / Soul---Free Funk": "Funk/Soul", "Funk / Soul---Funk": "Funk/Soul",
    "Funk / Soul---Gospel": "Gospel", "Funk / Soul---Neo Soul": "R&B",
    "Funk / Soul---New Jack Swing": "R&B", "Funk / Soul---P.Funk": "Funk/Soul",
    "Funk / Soul---Psychedelic": "Funk/Soul", "Funk / Soul---Rhythm & Blues": "R&B",
    "Funk / Soul---Soul": "Funk/Soul", "Funk / Soul---Swingbeat": "R&B",
    "Funk / Soul---UK Street Soul": "R&B",
    # Hip Hop
    "Hip Hop---Bass Music": "Hip-Hop", "Hip Hop---Boom Bap": "Hip-Hop",
    "Hip Hop---Bounce": "Hip-Hop", "Hip Hop---Britcore": "Hip-Hop",
    "Hip Hop---Cloud Rap": "Hip-Hop", "Hip Hop---Conscious": "Hip-Hop",
    "Hip Hop---Crunk": "Hip-Hop", "Hip Hop---Cut-up/DJ": "Hip-Hop",
    "Hip Hop---DJ Battle Tool": "Hip-Hop", "Hip Hop---Electro": "Electronic",
    "Hip Hop---G-Funk": "Hip-Hop", "Hip Hop---Gangsta": "Hip-Hop",
    "Hip Hop---Grime": "Hip-Hop", "Hip Hop---Hardcore Hip-Hop": "Hip-Hop",
    "Hip Hop---Horrorcore": "Hip-Hop", "Hip Hop---Instrumental": "Hip-Hop",
    "Hip Hop---Jazzy Hip-Hop": "Hip-Hop", "Hip Hop---Miami Bass": "Hip-Hop",
    "Hip Hop---Pop Rap": "Hip-Hop", "Hip Hop---Ragga HipHop": "Hip-Hop",
    "Hip Hop---RnB/Swing": "R&B", "Hip Hop---Screw": "Hip-Hop",
    "Hip Hop---Thug Rap": "Hip-Hop", "Hip Hop---Trap": "Trap",
    "Hip Hop---Trip Hop": "Ambient", "Hip Hop---Turntablism": "Hip-Hop",
    # Jazz
    "Jazz---Afro-Cuban Jazz": "Jazz", "Jazz---Afrobeat": "Jazz",
    "Jazz---Avant-garde Jazz": "Jazz", "Jazz---Big Band": "Jazz",
    "Jazz---Bop": "Jazz", "Jazz---Bossa Nova": "Jazz",
    "Jazz---Contemporary Jazz": "Jazz", "Jazz---Cool Jazz": "Jazz",
    "Jazz---Dixieland": "Jazz", "Jazz---Easy Listening": "Jazz",
    "Jazz---Free Improvisation": "Jazz", "Jazz---Free Jazz": "Jazz",
    "Jazz---Fusion": "Jazz", "Jazz---Gypsy Jazz": "Jazz",
    "Jazz---Hard Bop": "Jazz", "Jazz---Jazz-Funk": "Jazz",
    "Jazz---Jazz-Rock": "Jazz", "Jazz---Latin Jazz": "Jazz",
    "Jazz---Modal": "Jazz", "Jazz---Post Bop": "Jazz",
    "Jazz---Ragtime": "Jazz", "Jazz---Smooth Jazz": "Jazz",
    "Jazz---Soul-Jazz": "Jazz", "Jazz---Space-Age": "Jazz", "Jazz---Swing": "Jazz",
    # Latin
    "Latin---Afro-Cuban": "Latin", "Latin---Baião": "Latin",
    "Latin---Batucada": "Latin", "Latin---Beguine": "Latin",
    "Latin---Bolero": "Latin", "Latin---Boogaloo": "Latin",
    "Latin---Bossanova": "Latin", "Latin---Cha-Cha": "Latin",
    "Latin---Charanga": "Latin", "Latin---Compas": "Latin",
    "Latin---Cubano": "Latin", "Latin---Cumbia": "Cumbia",
    "Latin---Descarga": "Latin", "Latin---Forró": "Latin",
    "Latin---Guaguancó": "Latin", "Latin---Guajira": "Latin",
    "Latin---Guaracha": "Latin", "Latin---MPB": "Latin",
    "Latin---Mambo": "Latin", "Latin---Mariachi": "Latin",
    "Latin---Merengue": "Merengue", "Latin---Norteño": "Latin",
    "Latin---Nueva Cancion": "Latin", "Latin---Pachanga": "Latin",
    "Latin---Porro": "Latin", "Latin---Ranchera": "Latin",
    "Latin---Reggaeton": "Reggaetón", "Latin---Rumba": "Latin",
    "Latin---Salsa": "Latin", "Latin---Samba": "Latin",
    "Latin---Son": "Latin", "Latin---Son Montuno": "Latin",
    "Latin---Tango": "Tango", "Latin---Tejano": "Latin", "Latin---Vallenato": "Latin",
    # Non-Music — skip
    "Non-Music---Audiobook": None, "Non-Music---Comedy": None,
    "Non-Music---Dialogue": None, "Non-Music---Education": None,
    "Non-Music---Field Recording": None, "Non-Music---Interview": None,
    "Non-Music---Monolog": None, "Non-Music---Poetry": None,
    "Non-Music---Political": None, "Non-Music---Promotional": None,
    "Non-Music---Radioplay": None, "Non-Music---Religious": None,
    "Non-Music---Spoken Word": None,
    # Pop
    "Pop---Ballad": "Pop", "Pop---Bollywood": "World Music",
    "Pop---Bubblegum": "Pop", "Pop---Chanson": "Pop",
    "Pop---City Pop": "Pop", "Pop---Europop": "Pop",
    "Pop---Indie Pop": "Indie", "Pop---J-pop": "Pop",
    "Pop---K-pop": "K-Pop", "Pop---Kayōkyoku": "Pop",
    "Pop---Light Music": "Pop", "Pop---Music Hall": "Pop",
    "Pop---Novelty": "Pop", "Pop---Parody": "Pop",
    "Pop---Schlager": "Pop", "Pop---Vocal": "Pop",
    # Reggae
    "Reggae---Calypso": "Reggae", "Reggae---Dancehall": "Dancehall",
    "Reggae---Dub": "Reggae", "Reggae---Lovers Rock": "Reggae",
    "Reggae---Ragga": "Reggae", "Reggae---Reggae": "Reggae",
    "Reggae---Reggae-Pop": "Reggae", "Reggae---Rocksteady": "Reggae",
    "Reggae---Roots Reggae": "Reggae", "Reggae---Ska": "Reggae",
    "Reggae---Soca": "Reggae",
    # Rock
    "Rock---AOR": "Rock", "Rock---Acid Rock": "Rock",
    "Rock---Acoustic": "Acoustic", "Rock---Alternative Rock": "Alternative Rock",
    "Rock---Arena Rock": "Rock", "Rock---Art Rock": "Rock",
    "Rock---Atmospheric Black Metal": "Metal", "Rock---Avantgarde": "Rock",
    "Rock---Beat": "Rock", "Rock---Black Metal": "Metal",
    "Rock---Blues Rock": "Blues", "Rock---Brit Pop": "Indie",
    "Rock---Classic Rock": "Rock", "Rock---Coldwave": "Rock",
    "Rock---Country Rock": "Country", "Rock---Crust": "Punk",
    "Rock---Death Metal": "Metal", "Rock---Deathcore": "Metal",
    "Rock---Deathrock": "Rock", "Rock---Depressive Black Metal": "Metal",
    "Rock---Doo Wop": "Rock", "Rock---Doom Metal": "Metal",
    "Rock---Dream Pop": "Indie", "Rock---Emo": "Rock",
    "Rock---Ethereal": "Ambient", "Rock---Experimental": "Rock",
    "Rock---Folk Metal": "Metal", "Rock---Folk Rock": "Folk",
    "Rock---Funeral Doom Metal": "Metal", "Rock---Funk Metal": "Metal",
    "Rock---Garage Rock": "Rock", "Rock---Glam": "Rock",
    "Rock---Goregrind": "Metal", "Rock---Goth Rock": "Rock",
    "Rock---Gothic Metal": "Metal", "Rock---Grindcore": "Metal",
    "Rock---Grunge": "Grunge", "Rock---Hard Rock": "Hard Rock",
    "Rock---Hardcore": "Punk", "Rock---Heavy Metal": "Metal",
    "Rock---Indie Rock": "Indie", "Rock---Industrial": "Electronic",
    "Rock---Krautrock": "Rock", "Rock---Lo-Fi": "Lo-Fi",
    "Rock---Lounge": "Jazz", "Rock---Math Rock": "Progressive Rock",
    "Rock---Melodic Death Metal": "Metal", "Rock---Melodic Hardcore": "Punk",
    "Rock---Metalcore": "Metal", "Rock---Mod": "Rock",
    "Rock---Neofolk": "Folk", "Rock---New Wave": "Rock",
    "Rock---No Wave": "Rock", "Rock---Noise": "Rock",
    "Rock---Noisecore": "Metal", "Rock---Nu Metal": "Metal",
    "Rock---Oi": "Punk", "Rock---Parody": "Rock",
    "Rock---Pop Punk": "Punk", "Rock---Pop Rock": "Pop",
    "Rock---Pornogrind": "Metal", "Rock---Post Rock": "Rock",
    "Rock---Post-Hardcore": "Punk", "Rock---Post-Metal": "Metal",
    "Rock---Post-Punk": "Rock", "Rock---Power Metal": "Metal",
    "Rock---Power Pop": "Pop", "Rock---Power Violence": "Punk",
    "Rock---Prog Rock": "Progressive Rock", "Rock---Progressive Metal": "Metal",
    "Rock---Psychedelic Rock": "Rock", "Rock---Psychobilly": "Rock",
    "Rock---Pub Rock": "Rock", "Rock---Punk": "Punk",
    "Rock---Rock & Roll": "Rock", "Rock---Rockabilly": "Rock",
    "Rock---Shoegaze": "Indie", "Rock---Ska": "Reggae",
    "Rock---Sludge Metal": "Metal", "Rock---Soft Rock": "Pop",
    "Rock---Southern Rock": "Rock", "Rock---Space Rock": "Rock",
    "Rock---Speed Metal": "Metal", "Rock---Stoner Rock": "Rock",
    "Rock---Surf": "Rock", "Rock---Symphonic Rock": "Rock",
    "Rock---Technical Death Metal": "Metal", "Rock---Thrash": "Metal",
    "Rock---Twist": "Rock", "Rock---Viking Metal": "Metal", "Rock---Yé-Yé": "Pop",
    # Stage & Screen
    "Stage & Screen---Musical": "Musical Theatre",
    "Stage & Screen---Score": "Film Score",
    "Stage & Screen---Soundtrack": "Film Score",
    "Stage & Screen---Theme": "Film Score",
}

# 1. GLOBAL VARIABLES FOR AI MODELS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yamnet_model       = None   # tf.saved_model.load — avoids hub.load TF op conflict
mood_model         = None
mood_encoder       = None
instrument_model   = None
instrument_encoder = None
beatnet_estimator  = None
_beatnet_pool      = None
DISCOGS_PB     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'discogs-effnet-bs64-1.pb')
DISCOGS_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'discogs_predict.py')

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

# ─── YAMNet class indices (from AudioSet ontology) ───────────────────────────
# Vocals
_YAMNET_SINGING     = 24   # "Singing"
_YAMNET_CHOIR       = 25   # "Choir"
_YAMNET_RAPPING     = 31   # "Rapping"
_YAMNET_VOCAL_MUSIC = 249  # "Vocal music"
_YAMNET_SPEECH      = 0    # "Speech"

# Instruments — grouped by family, use max() across related classes
_YAMNET_INSTRUMENTS = {
    "Piano":           [148, 149],          # Piano, Electric piano
    "Organ":           [150, 151, 152],     # Organ, Electronic organ, Hammond organ
    "Synthesizer":     [153],               # Synthesizer
    "Guitar":          [135, 138, 139],     # Guitar, Acoustic guitar, Slide guitar
    "Electric Guitar": [136],              # Electric guitar
    "Bass":            [137, 189],          # Bass guitar, Double bass
    "Drums":           [157, 158, 159, 160, 163],  # Drum kit, machine, snare, bass drum
    "Strings":         [184, 185, 186, 188], # Bowed string, String section, Violin, Cello
    "Brass":           [182],              # Trumpet
    "Flute":           [191],              # Flute
    "Saxophone":       [192],              # Saxophone
}
_YAMNET_INST_THRESHOLD = 0.05  # 25x above random baseline (1/521 ≈ 0.002)


def detect_instruments_yamnet(yamnet_class_scores: np.ndarray) -> list:
    """Detect instruments directly from YAMNet AudioSet class scores.

    Uses YAMNet's built-in instrument knowledge (trained on 2M+ AudioSet clips)
    instead of our smaller trained classifier. Much more accurate.

    Args:
        yamnet_class_scores: (N_frames, 521) YAMNet class scores per frame

    Returns:
        List of {"instrument": name, "confidence": score} sorted by confidence
    """
    mean_scores = np.mean(yamnet_class_scores, axis=0)   # (521,)
    results = []
    for inst_name, indices in _YAMNET_INSTRUMENTS.items():
        score = float(max(mean_scores[i] for i in indices))
        if score >= _YAMNET_INST_THRESHOLD:
            results.append({"instrument": inst_name, "confidence": round(score, 3)})
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results

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

    VOCAL_THRESHOLD = 0.04   # lowered: mixed vocals in music score 0.04-0.07

    print(f"Vocal scores — singing:{singing_score:.3f} vocal_music:{vocal_score:.3f} "
          f"choir:{choir_score:.3f} rap:{rap_score:.3f} total:{total_vocal:.3f}")
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


def get_use_cases(genre: str, moods: list, energy: int, vocals: str,
                  bpm: int = 120, time_sig: str = '4/4') -> list[str]:
    """Derive sync licensing use case tags from all available metadata.

    Uses genre, mood, energy, vocals, BPM and time signature for richer
    placement suggestions. Returns up to 4 tags ordered by confidence.
    """
    g         = genre.lower()
    all_moods = {m["mood"].lower() for m in moods}
    tags      = []

    # ── Film & TV ─────────────────────────────────────────────────────────────
    # Triumphant removed — it belongs in ads/sports, not film. Film needs
    # genuinely cinematic moods: epic, mysterious, suspense, dark, tense.
    film_genres = {'film score', 'cinematic', 'classical', 'baroque', 'opera', 'ambient'}
    film_moods  = {'epic', 'mysterious', 'suspense', 'dark', 'tense', 'atmospheric'}
    if g in film_genres or len(all_moods & film_moods) >= 1:
        tags.append('Film & TV')

    # ── Trailer / Epic ────────────────────────────────────────────────────────
    # Requires 'epic' specifically (not triumphant) and very high energy (8+).
    # Prevents pop/folk songs with triumphant mood from getting this tag.
    if 'epic' in all_moods and energy >= 8:
        tags.append('Trailer / Epic')

    # ── Advertising / Commercial ──────────────────────────────────────────────
    ad_moods = {'uplifting', 'happy', 'energetic', 'inspiring', 'playful', 'triumphant', 'groovy'}
    if len(all_moods & ad_moods) >= 1 and energy >= 4:
        tags.append('Advertising')

    # ── Corporate / Brand ─────────────────────────────────────────────────────
    corp_genres = {'corporate', 'acoustic', 'pop', 'indie'}
    corp_moods  = {'uplifting', 'happy', 'inspiring', 'calm', 'playful'}
    if g in corp_genres or len(all_moods & corp_moods) >= 1:
        tags.append('Corporate / Brand')

    # ── Sports / Action ───────────────────────────────────────────────────────
    # Triumphant added here — it fits sports highlights and action content.
    if energy >= 8 or (energy >= 7 and len(all_moods & {'aggressive', 'energetic', 'angry', 'triumphant'}) >= 1):
        tags.append('Sports / Action')

    # ── Gaming ────────────────────────────────────────────────────────────────
    game_genres = {'electronic', 'techno', 'dubstep', 'drum & bass', 'edm', 'trance',
                   'house', 'metal', 'rock', 'synthwave', 'hip-hop', 'trap'}
    game_moods  = {'epic', 'aggressive', 'mysterious', 'dark', 'energetic', 'tense'}
    if (g in game_genres and energy >= 6) or len(all_moods & game_moods) >= 2:
        tags.append('Gaming')

    # ── Study / Focus / Background ────────────────────────────────────────────
    chill_genres = {'lo-fi', 'ambient', 'new age', 'corporate', 'classical'}
    if g in chill_genres or (energy <= 4 and vocals == 'Instrumental'):
        tags.append('Study / Focus')

    # ── Meditation / Wellness ─────────────────────────────────────────────────
    if (g in {'ambient', 'new age'} and energy <= 4) or \
       (energy <= 3 and vocals == 'Instrumental'):
        tags.append('Meditation / Wellness')

    # ── Romance / Wedding ─────────────────────────────────────────────────────
    rom_moods = {'romantic', 'nostalgic', 'melancholic', 'hopeful', 'dreamy'}
    if len(all_moods & rom_moods) >= 1 and energy <= 7:
        tags.append('Romance / Wedding')

    # ── Waltz / Ballroom ──────────────────────────────────────────────────────
    if time_sig == '3/4' and energy <= 6:
        tags.append('Waltz / Ballroom')

    # ── Documentary / Nature ──────────────────────────────────────────────────
    doc_genres = {'folk', 'world music', 'classical', 'ambient', 'acoustic', 'new age'}
    if g in doc_genres or 'atmospheric' in all_moods:
        tags.append('Documentary')

    # ── Social Media / Content Creation ──────────────────────────────────────
    social_genres = {'pop', 'indie', 'hip-hop', 'r&b', 'edm', 'trap', 'afrobeats', 'k-pop',
                     'reggaeton', 'dancehall', 'house'}
    if g in social_genres and energy >= 5:
        tags.append('Social Media / Content')

    # ── Dance / Club ─────────────────────────────────────────────────────────
    dance_genres = {'house', 'techno', 'edm', 'trance', 'drum & bass', 'dubstep', 'reggaeton'}
    if g in dance_genres and bpm >= 120:
        tags.append('Dance / Club')

    # Deduplicate, keep top 4
    seen, result = set(), []
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

# 🔥 THIS IS THE FIX: Load models AFTER the server port opens
def _load_models_sync():
    """Load all models synchronously — runs in a thread pool so the event loop
    (and therefore /health) stays responsive during the 60-90 second load."""
    global yamnet_model, mood_model, mood_encoder
    global instrument_model, instrument_encoder
    global _models_ready
    print("🚪 Port is open! Now loading AI brains in a background thread...")

    # ── YAMNet via tf.saved_model.load (pre-downloaded at build time) ──────────
    # Using tf.saved_model.load instead of hub.load avoids the hub runtime
    # loading a TF1-compat layer that conflicts with essentia's TF op registry.
    yamnet_model = tf.saved_model.load(os.path.join(BASE_DIR, 'yamnet_model'))
    print("✅ YAMNet loaded via tf.saved_model.load")

    # Discogs-EfficientNet runs in a subprocess (discogs_predict.py) per request.
    # This avoids the ALREADY_EXISTS: Op with name Bitcast crash that occurs when
    # essentia's bundled old TF and pip-tensorflow are both loaded in the same process.
    print(f"✅ Discogs-EfficientNet will run via subprocess (discogs_predict.py)")

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

    print("✅ All AI Brains successfully loaded and ready for traffic!")

    # ── BeatNet time signature estimator ──
    global beatnet_estimator, _beatnet_pool
    try:
        from BeatNet.BeatNet import BeatNet
        beatnet_estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)
        _beatnet_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        print("✅ BeatNet loaded — time signature detection active (5s timeout per request)")
    except Exception as e:
        print(f"⚠️  BeatNet unavailable ({e}) — falling back to autocorrelation")

    _models_ready = True
    print("✅ _models_ready = True — /health will now return 200")


@app.on_event("startup")
async def load_all_models():
    """Kick off model loading in a thread so the HTTP server (and /health) stays
    responsive immediately. Cloud Run startup probe can reach /health right away
    and will keep getting 503 until models finish, then 200."""
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _load_models_sync)


@app.get("/health")
async def health():
    """Startup health probe for Cloud Run.

    Returns 200 only after all AI models have finished loading.
    Cloud Run startup probe uses this to avoid routing traffic to
    a booting instance — eliminates 503s during restarts/deployments.
    """
    if not _models_ready:
        raise HTTPException(status_code=503, detail="Models still loading")
    return {"status": "ok"}

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

        # ── Time signature — BeatNet with 5s hard timeout, autocorrelation fallback
        # BeatNet's DBN can hang on some audio files without a timeout.
        # We run it in a dedicated thread pool and abort after 5 seconds.
        if beatnet_estimator is not None and _beatnet_pool is not None:
            try:
                future = _beatnet_pool.submit(beatnet_estimator.process, temp_path)
                beat_output = future.result(timeout=5)
                if beat_output is not None and len(beat_output) > 0:
                    max_beat = int(np.max(beat_output[:, 1]))
                    time_signature = '3/4' if max_beat == 3 else ('6/8' if max_beat == 6 else '4/4')
                else:
                    time_signature = detect_time_signature(y22, SR_LIBROSA, HOP_LENGTH)
            except concurrent.futures.TimeoutError:
                print("⚠️  BeatNet timed out — using autocorrelation fallback")
                time_signature = detect_time_signature(y22, SR_LIBROSA, HOP_LENGTH)
            except Exception:
                time_signature = detect_time_signature(y22, SR_LIBROSA, HOP_LENGTH)
        else:
            time_signature = detect_time_signature(y22, SR_LIBROSA, HOP_LENGTH)

        # ── Resample to 16 kHz for YAMNet ────────────────────────────────────
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

        # ── YAMNet — class scores for vocal/instrument + embeddings for mood ──
        class_scores, embeddings, _ = yamnet_model(y16)
        class_scores_np = class_scores.numpy()   # (N_frames, 521)
        emb_np          = embeddings.numpy()
        yamnet_mean     = np.mean(emb_np, axis=0)  # (1024,)

        # Vocal detection uses raw class scores before any pooling
        vocals, vocals_confidence = detect_vocals(class_scores_np)

        # ── Mood + instrument both use 1024-dim YAMNet mean embedding ───────────
        X_mood = yamnet_mean[np.newaxis, :]   # (1, 1024)
        mood_preds = mood_model.predict(X_mood, verbose=0)

        # ── Instrument detection via YAMNet AudioSet classes ─────────────────
        detected_instruments = detect_instruments_yamnet(class_scores_np)

        # ── Genre via Discogs-EfficientNet subprocess (essentia isolated from pip-TF) ──
        try:
            _env = {**os.environ, 'TF_CPP_MIN_LOG_LEVEL': '3', 'TF_ENABLE_ONEDNN_OPTS': '0'}
            _proc = subprocess.run(
                ['python3', DISCOGS_SCRIPT, temp_path, DISCOGS_PB],
                capture_output=True, text=True, timeout=30, env=_env,
            )
            if _proc.returncode == 0 and _proc.stdout.strip():
                genre_probs = np.array(json.loads(_proc.stdout.strip()), dtype=np.float32)

                genre_scores: dict[str, float] = {}
                for _i, _prob in enumerate(genre_probs.tolist()):
                    _label = DISCOGS400_CLASSES[_i]
                    _clean = DISCOGS400_TO_GENRE.get(_label)
                    if _clean is None:
                        continue
                    if _clean not in genre_scores or _prob > genre_scores[_clean]:
                        genre_scores[_clean] = float(_prob)

                ranked = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
                top10_raw = sorted(zip(genre_probs.tolist(), DISCOGS400_CLASSES), reverse=True)[:10]
                print(f"Top 10 Discogs labels: {[(lbl, round(p,4)) for p,lbl in top10_raw]}")
                print(f"Top 5 genre candidates: {[(g, round(s,4)) for g,s in ranked[:5]]}")
            else:
                print(f"⚠️  Discogs subprocess failed (rc={_proc.returncode}): {_proc.stderr[:300]}")
                ranked = [("Unknown", 0.0)]
        except subprocess.TimeoutExpired:
            print("⚠️  Discogs subprocess timed out after 30s")
            ranked = [("Unknown", 0.0)]
        except Exception as _e:
            print(f"⚠️  Discogs genre prediction error: {_e}")
            ranked = [("Unknown", 0.0)]

        primary_genre        = ranked[0][0]
        primary_genre_conf   = ranked[0][1]
        secondary_genre      = ranked[1][0] if len(ranked) > 1 else primary_genre
        secondary_genre_conf = ranked[1][1] if len(ranked) > 1 else primary_genre_conf
        tertiary_genre       = ranked[2][0] if len(ranked) > 2 else secondary_genre
        tertiary_genre_conf  = ranked[2][1] if len(ranked) > 2 else secondary_genre_conf

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
        use_cases        = get_use_cases(clean_genre, clean_moods, energy_level, vocals,
                                          bpm=bpm, time_sig=time_signature)

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