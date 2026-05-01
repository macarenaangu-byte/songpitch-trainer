#!/bin/bash
# run_training_pipeline.sh
# Full retraining pipeline: augment → extract features → train models
# Run from: /Users/macarena.nadeau/Desktop/songpitch-trainer/
# Usage: bash run_training_pipeline.sh

set -e
cd "$(dirname "$0")"

# ── Keep the Mac awake for the entire training run ──────────────────────────
# caffeinate -dims: prevent display sleep (-d), idle sleep (-i),
# system sleep (-m), and disk sleep (-s)
caffeinate -dims &
CAFFEINATE_PID=$!
trap "kill $CAFFEINATE_PID 2>/dev/null; echo '☕ caffeinate stopped'" EXIT
echo "☕ caffeinate started (PID $CAFFEINATE_PID) — laptop will stay awake"
echo ""

echo "============================================================"
echo "  SongPitch Genre Model — Full Training Pipeline"
echo "============================================================"
echo ""

# Step 1: Augment underrepresented genres
echo "STEP 1/3 — Augmenting audio data..."
echo "  (genres < 200 files get 7 variants, 200-499 get 2 variants)"
echo ""
/usr/bin/python3 augment_audio.py
echo ""
echo "✅ Augmentation complete"
echo ""

# Step 2: Extract YAMNet features from ALL training data
echo "STEP 2/3 — Extracting YAMNet features..."
echo "  (this takes 30-90 min depending on dataset size)"
echo ""
# Delete old checkpoint so we do a full fresh extraction with all new genres
if [ -f yamnet_extraction_checkpoint.pkl ]; then
  echo "  Removing old extraction checkpoint to include new genres..."
  rm yamnet_extraction_checkpoint.pkl
fi
/usr/bin/python3 extract_yamnet_features.py
echo ""
echo "✅ Feature extraction complete"
echo ""

# Step 3: Train genre + mood classifiers
echo "STEP 3/3 — Training classifiers..."
echo "  (genre model + mood model, up to 150 epochs each with early stopping)"
echo ""
/usr/bin/python3 train_yamnet_classifier.py
echo ""
echo "✅ Training complete"
echo ""

echo "============================================================"
echo "  Pipeline finished!"
echo "  Files to deploy to Render:"
echo "    yamnet_genre_model.keras"
echo "    yamnet_genre_model_encoder.pkl"
echo "    yamnet_mood_model.keras"
echo "    yamnet_mood_model_encoder.pkl"
echo ""
echo "  Next: git add those 4 files + main.py → git commit → git push"
echo "  Render will auto-redeploy (~2 min)"
echo "============================================================"
