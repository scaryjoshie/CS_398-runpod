"""Single Inference page — upload an audio file and get raw model output."""

import sys
import io
import json
import os
import tempfile
from pathlib import Path

# Ensure project root is on path for imports
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from ui_common import show_gpu_sidebar

st.set_page_config(page_title="Single Inference", page_icon="🎙️", layout="wide")
show_gpu_sidebar()
st.title("Single Inference")
st.markdown("Upload an audio file **or record from your microphone** and see the raw output from any loaded model.")

# ---------------------------------------------------------------------------
# Audio normalization — converts ANY audio format to 16kHz mono WAV bytes
# ---------------------------------------------------------------------------

def normalize_audio_to_wav(raw_bytes: bytes, original_suffix: str = ".wav") -> bytes | None:
    """Convert arbitrary audio bytes to 16kHz mono PCM WAV.

    Handles MP3, WebM, OGG, FLAC, etc. via librosa (which uses ffmpeg).
    Returns WAV bytes, or None on failure.
    """
    import soundfile as sf
    import librosa

    # Write raw bytes to temp file with original extension so librosa/ffmpeg
    # can detect the codec
    with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        signal, sr = librosa.load(tmp_path, sr=16000, mono=True)
        import numpy as np
        st.session_state["_audio_debug"] = {
            "duration_s": len(signal) / sr,
            "samples": len(signal),
            "sr": sr,
            "max_amplitude": float(np.abs(signal).max()),
            "is_silence": float(np.abs(signal).max()) < 0.001,
        }
        wav_buf = io.BytesIO()
        sf.write(wav_buf, signal, 16000, format="WAV", subtype="PCM_16")
        return wav_buf.getvalue()
    except Exception as e:
        st.error(f"Failed to decode audio: {e}")
        st.info("Make sure ffmpeg is installed (`apt install ffmpeg`).")
        return None
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Display helpers
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

raw_audio_bytes = None
audio_suffix = ".wav"

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "flac", "ogg"],
        help="Supported formats: WAV, MP3, FLAC, OGG. Will be converted to 16kHz mono internally.",
    )
    if uploaded_file is not None:
        raw_audio_bytes = uploaded_file.getvalue()
        audio_suffix = Path(uploaded_file.name).suffix or ".wav"

with tab_mic:
    recorded = st.audio_input("Click to record from your microphone")
    if recorded is not None:
        raw_audio_bytes = recorded.getvalue()
        # st.audio_input returns WAV in Streamlit >=1.33
        audio_suffix = ".wav"

# ---------------------------------------------------------------------------
# Normalize + Inference
# ---------------------------------------------------------------------------

if raw_audio_bytes is not None:
    # Debug info
    with st.expander("🐛 Audio debug info"):
        st.text(f"Raw bytes size: {len(raw_audio_bytes):,} bytes")
        st.text(f"File suffix: {audio_suffix}")
        st.text(f"First 16 bytes (hex): {raw_audio_bytes[:16].hex()}")
        # Check if it starts with RIFF (WAV header)
        if raw_audio_bytes[:4] == b'RIFF':
            st.text("Format: WAV (RIFF header detected)")
        elif raw_audio_bytes[:4] == b'\x1aE\xdf\xa3':
            st.text("Format: WebM/Matroska header detected")
        elif raw_audio_bytes[:4] == b'OggS':
            st.text("Format: OGG header detected")
        elif raw_audio_bytes[:3] == b'ID3' or raw_audio_bytes[:2] == b'\xff\xfb':
            st.text("Format: MP3 header detected")
        else:
            st.text(f"Format: Unknown (header: {raw_audio_bytes[:4]})")

    # Normalize ALL audio to 16kHz mono WAV — this fixes MP3, WebM, OGG, etc.
    wav_bytes = normalize_audio_to_wav(raw_audio_bytes, audio_suffix)

    if wav_bytes is not None:
        # Show conversion debug
        debug = st.session_state.get("_audio_debug", {})
        if debug:
            with st.expander("🐛 Conversion debug info"):
                st.text(f"Duration: {debug.get('duration_s', 0):.2f}s")
                st.text(f"Samples: {debug.get('samples', 0)} @ {debug.get('sr', 0)} Hz")
                st.text(f"Max amplitude: {debug.get('max_amplitude', 0):.6f}")
                if debug.get("is_silence"):
                    st.warning("⚠️ Audio appears to be silence (max amplitude < 0.001)")
                st.text(f"Output WAV size: {len(wav_bytes):,} bytes")

        # Audio playback (now guaranteed to be valid WAV)
        st.audio(wav_bytes, format="audio/wav")

        # Save WAV to temp file for model inference
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(wav_bytes)
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
