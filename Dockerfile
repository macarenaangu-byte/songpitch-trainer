# ─── Stage 1: Build dependencies ──────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# System deps for librosa / audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libgomp1 \
    build-essential \
    libfftw3-dev \
    libsamplerate0-dev \
    libyaml-dev \
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
    libfftw3-3 \
    libsamplerate0 \
    libyaml-0-2 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code and model files
COPY main.py .
COPY requirements.txt .

# Discogs EfficientNet frozen graph
COPY discogs-effnet-bs64-1.pb .

# Genre is now predicted directly by Essentia + discogs-effnet-bs64-1.pb (400 Discogs classes)

# Mood model
COPY yamnet_mood_model.h5 .
COPY yamnet_mood_model_encoder.pkl .

# Instrument model
COPY instrument_model.h5 .
COPY instrument_model_encoder.pkl .

# Cloud Run sets PORT env var (default 8080)
ENV PORT=8080

# Use a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Start the FastAPI server — binds to $PORT as required by Cloud Run
CMD uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 120
