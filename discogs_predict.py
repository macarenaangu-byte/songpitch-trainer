#!/usr/bin/env python3
"""Discogs-EfficientNet genre inference, isolated from the main FastAPI process.

This script is intentionally kept in a separate process. The main process imports
pip-tensorflow (2.x). Importing essentia.standard in that same process loads essentia's
bundled old TF runtime and triggers a fatal C++ op-registry conflict
(ALREADY_EXISTS: Op with name Bitcast). Running in a subprocess gives each TF runtime
its own process memory space and its own op registry — no conflict possible.

Usage:  python3 discogs_predict.py <audio_path> <model_pb_path>
Output: JSON array of 400 sigmoid probabilities printed to stdout.
"""
import sys
import json
import numpy as np


def main():
    if len(sys.argv) < 3:
        sys.exit("Usage: discogs_predict.py <audio_path> <model_pb_path>")

    audio_path = sys.argv[1]
    model_path = sys.argv[2]

    import essentia.standard as es

    audio = es.MonoLoader(filename=audio_path, sampleRate=16000)()
    predictor = es.TensorflowPredictEffnetDiscogs(
        graphFilename=model_path,
        output='PartitionedCall:0',
    )
    patch_preds = predictor(audio)      # (N_patches, 400) sigmoid scores
    genre_probs = np.mean(patch_preds, axis=0).tolist()
    print(json.dumps(genre_probs))


if __name__ == '__main__':
    main()
