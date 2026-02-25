"""Single Inference page — upload an audio file and get raw model output."""

import sys
import io
import json
import tempfile
from pathlib import Path

# Ensure project root is on path for imports
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Single Inference", page_icon="🎙️", layout="wide")
st.title("🎙️ Single Inference")
st.markdown("Upload an audio file **or record from your microphone** and see the raw output from any loaded model.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def display_probabilities(result: dict):
    """Display bar chart for probability-type model output."""
    labels = result["labels"]
    scores = result["scores"]

    st.markdown(f"**Top prediction:** `{result['top_label']}` ({result['top_score']:.3f})")

    # Sort by score descending for the chart
    pairs = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
    sorted_labels, sorted_scores = zip(*pairs)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2ecc71" if l == result["top_label"] else "#3498db" for l in sorted_labels]
    bars = ax.barh(range(len(sorted_labels)), sorted_scores, color=colors)
    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score")
    ax.set_title("Emotion Probabilities")
    ax.invert_yaxis()

    # Add score text on bars
    for bar, score in zip(bars, sorted_scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def display_dimensional(result: dict):
    """Display dimensional output (arousal, dominance, valence)."""
    dims = result["dimensions"]

    cols = st.columns(3)
    for col, (dim_name, value) in zip(cols, dims.items()):
        with col:
            # Color code: low=blue, mid=gray, high=red
            st.metric(label=dim_name.capitalize(), value=f"{value:.3f}")
            st.progress(min(max(value, 0.0), 1.0))

    # Also show as a bar chart
    fig, ax = plt.subplots(figsize=(6, 3))
    dim_names = list(dims.keys())
    dim_values = list(dims.values())
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    ax.barh(dim_names, dim_values, color=colors)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Value")
    ax.set_title("Dimensional Emotion Values")
    for i, v in enumerate(dim_values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

# Model selection
if "model_registry" not in st.session_state:
    st.error("Model registry not initialized. Please run from the main app.py entry point.")
    st.stop()

model_registry = st.session_state["model_registry"]
model_name = st.selectbox("Select Model", list(model_registry.keys()))

# ---------------------------------------------------------------------------
# Audio input — upload OR microphone
# ---------------------------------------------------------------------------

tab_upload, tab_mic = st.tabs(["📁 Upload File", "🎤 Record from Mic"])

audio_bytes = None
audio_name = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "flac", "ogg"],
        help="Supported formats: WAV, MP3, FLAC, OGG. Will be converted to 16kHz mono internally.",
    )
    if uploaded_file is not None:
        audio_bytes = uploaded_file.getvalue()
        audio_name = uploaded_file.name

with tab_mic:
    recorded = st.audio_input("Click to record from your microphone")
    if recorded is not None:
        # Browser records as WebM/OGG, not WAV. Convert to WAV via librosa/soundfile.
        try:
            import soundfile as sf
            import librosa

            raw_bytes = recorded.getvalue()
            # Write browser audio to a temp file with original format
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_raw:
                tmp_raw.write(raw_bytes)
                tmp_raw_path = tmp_raw.name

            # Load with librosa (handles webm/ogg via ffmpeg/soundfile)
            signal, sr = librosa.load(tmp_raw_path, sr=16000, mono=True)

            # Write back as proper WAV
            wav_buf = io.BytesIO()
            sf.write(wav_buf, signal, 16000, format="WAV", subtype="PCM_16")
            audio_bytes = wav_buf.getvalue()
            audio_name = "recording.wav"

            import os
            os.unlink(tmp_raw_path)
        except Exception as e:
            st.error(f"Failed to convert mic recording: {e}")
            st.info("Make sure ffmpeg is installed (`apt install ffmpeg`) for mic recording support.")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

if audio_bytes is not None:
    # Audio playback
    st.audio(audio_bytes)

    # Save to temp file for model inference
    suffix = Path(audio_name).suffix if audio_name else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # Run inference
    if st.button("🔍 Run Inference", type="primary"):
        model = model_registry[model_name]

        with st.spinner(f"Loading {model_name}..." if not model.is_loaded else f"Running {model_name}..."):
            model.ensure_loaded()
            result = model.predict(tmp_path)

        st.success("Inference complete!")

        # Display based on output type
        if model.output_type == "probabilities":
            display_probabilities(result)
        elif model.output_type == "dimensional":
            display_dimensional(result)
        else:
            st.warning(f"Unknown output type: {model.output_type}")

        # Raw JSON output
        with st.expander("🔧 Raw model output (JSON)"):
            st.json(result)
