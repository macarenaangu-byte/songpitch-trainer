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
COPY requirements.txt .

# Discogs EfficientNet frozen graph
COPY discogs-effnet-bs64-1.pb .

# Feature scalers
COPY feature_scaler.pkl .

# Stage 1 model
COPY stage1_model.h5 .
COPY stage1_model_encoder.pkl .

# Stage 2 models — one per genre category
COPY stage2_Ambient_Chill_model.h5 .
COPY stage2_Ambient_Chill_model_encoder.pkl .
COPY stage2_Classical_Cinematic_model.h5 .
COPY stage2_Classical_Cinematic_model_encoder.pkl .
COPY stage2_Electronic_model.h5 .
COPY stage2_Electronic_model_encoder.pkl .
COPY stage2_Folk_Country_Roots_model.h5 .
COPY stage2_Folk_Country_Roots_model_encoder.pkl .
COPY stage2_HipHop_Urban_model.h5 .
COPY stage2_HipHop_Urban_model_encoder.pkl .
COPY stage2_Jazz_Blues_model.h5 .
COPY stage2_Jazz_Blues_model_encoder.pkl .
COPY stage2_Latin_model.h5 .
COPY stage2_Latin_model_encoder.pkl .
COPY stage2_Pop_Indie_model.h5 .
COPY stage2_Pop_Indie_model_encoder.pkl .
COPY stage2_Rock_Metal_model.h5 .
COPY stage2_Rock_Metal_model_encoder.pkl .
COPY stage2_Theatrical_model.h5 .
COPY stage2_Theatrical_model_encoder.pkl .

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
