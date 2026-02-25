#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="/workspace/data"
mkdir -p "$DATA_DIR"

# ─────────────────────────────────────────────
# 1. RAVDESS — direct download from Zenodo
# ─────────────────────────────────────────────
RAVDESS_DIR="$DATA_DIR/RAVDESS"
RAVDESS_ZIP="$DATA_DIR/Audio_Speech_Actors_01-24.zip"
RAVDESS_URL="https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"

if [ -d "$RAVDESS_DIR" ] && [ "$(ls -A "$RAVDESS_DIR" 2>/dev/null)" ]; then
    echo "✅ RAVDESS already exists at $RAVDESS_DIR — skipping"
else
    echo "⬇️  Downloading RAVDESS from Zenodo (~215 MB)..."
    wget -q --show-progress -O "$RAVDESS_ZIP" "$RAVDESS_URL"
    echo "📦 Extracting..."
    mkdir -p "$RAVDESS_DIR"
    unzip -q "$RAVDESS_ZIP" -d "$RAVDESS_DIR"
    rm "$RAVDESS_ZIP"
    echo "✅ RAVDESS ready at $RAVDESS_DIR"
fi

# ─────────────────────────────────────────────
# 2. ESD — requires Kaggle CLI or manual download
# ─────────────────────────────────────────────
ESD_DIR="$DATA_DIR/ESD"

if [ -d "$ESD_DIR" ] && [ "$(ls -A "$ESD_DIR" 2>/dev/null)" ]; then
    echo "✅ ESD already exists at $ESD_DIR — skipping"
else
    # Try kaggle CLI first
    if command -v kaggle &> /dev/null; then
        echo "⬇️  Downloading ESD from Kaggle..."
        kaggle datasets download -d nguyenthanhlim/emotional-speech-dataset-esd -p "$DATA_DIR"
        echo "📦 Extracting..."
        mkdir -p "$ESD_DIR"
        unzip -q "$DATA_DIR/emotional-speech-dataset-esd.zip" -d "$ESD_DIR"
        rm "$DATA_DIR/emotional-speech-dataset-esd.zip"
        echo "✅ ESD ready at $ESD_DIR"
    else
        echo ""
        echo "⚠️  ESD requires manual download (no kaggle CLI found)."
        echo ""
        echo "   Option A — Install kaggle CLI:"
        echo "     pip install kaggle"
        echo "     # Put your kaggle.json at ~/.kaggle/kaggle.json"
        echo "     # Then re-run this script"
        echo ""
        echo "   Option B — Manual download:"
        echo "     1. Go to: https://www.kaggle.com/datasets/nguyenthanhlim/emotional-speech-dataset-esd"
        echo "     2. Download and extract to: $ESD_DIR"
        echo "     3. Structure should be: $ESD_DIR/0011/Angry/test/*.wav"
        echo ""
    fi
fi

echo ""
echo "── Summary ──"
[ -d "$RAVDESS_DIR" ] && echo "RAVDESS: $RAVDESS_DIR ($(find "$RAVDESS_DIR" -name '*.wav' 2>/dev/null | wc -l) wav files)" || echo "RAVDESS: ❌ not found"
[ -d "$ESD_DIR" ] && echo "ESD:     $ESD_DIR ($(find "$ESD_DIR" -name '*.wav' 2>/dev/null | wc -l) wav files)" || echo "ESD:     ❌ not found"
