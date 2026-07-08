# ─── Stage 1: Build dependencies ──────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# System deps for librosa / audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages into a separate prefix so we can copy them cleanly
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt



# ─── Stage 2: Runtime image ───────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Runtime system libs only (no build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code and model files
COPY main.py .
COPY discogs_predict.py .
COPY requirements.txt .

# Discogs-EfficientNet frozen graph (loaded in subprocess via essentia — never in main process)
COPY discogs-effnet-bs64-1.pb .
# YAMNet SavedModel bundled in repo
COPY yamnet_model ./yamnet_model

# YAMNet genre model (secondary/tertiary classification — trained on our taxonomy)
COPY yamnet_genre_model.h5 .
COPY yamnet_genre_model_encoder.pkl .

# Mood model
COPY yamnet_mood_model.h5 .
COPY yamnet_mood_model_encoder.pkl .

# Instrument model
COPY instrument_model.h5 .
COPY instrument_model_encoder.pkl .

# Cloud Run sets PORT env var (default 8080)
ENV PORT=8080
# Disable oneDNN (Intel Deep Neural Network library) in the main TF process.
# When oneDNN is on, it enables FPE exception trapping in native threads; this
# causes SIGFPE (signal 8) to kill uvicorn when librosa's FFT encounters a
# denormal value during beat/tempo estimation. Must be set before TF imports.
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_CPP_MIN_LOG_LEVEL=2

# Use a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Start the FastAPI server — binds to $PORT as required by Cloud Run
CMD uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 120
