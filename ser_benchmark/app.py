"""SER Benchmarking Suite — Main Entry Point.

Run with:
    cd ser_benchmark
    streamlit run app.py --server.port 8501

Then access via SSH tunnel:
    ssh -L 8501:localhost:8501 root@<runpod-ip> -p <runpod-ssh-port>
    Open http://localhost:8501 in your browser.
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure the project root is on the Python path so that
# `from models import ...` and `from datasets import ...` work
# from both the main app and the pages/ subdirectory.
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import MODEL_REGISTRY
from datasets import DATASET_REGISTRY

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "model_registry" not in st.session_state:
    st.session_state["model_registry"] = MODEL_REGISTRY

if "dataset_registry" not in st.session_state:
    st.session_state["dataset_registry"] = DATASET_REGISTRY

# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SER Benchmark Suite",
    page_icon="🎧",
    layout="wide",
)

st.title("🎧 SER Benchmarking Suite")
st.markdown("""
Welcome to the Speech Emotion Recognition benchmarking suite.

### Pages

- **🎙️ Single Inference** — Upload an audio file and get the raw output from any model
- **📊 Benchmark** — Run a model against a dataset and view the confusion matrix + metrics
- **🔬 Comparison** — Subtract two benchmark runs to isolate steering quality from classifier bias

---

### Loaded Models
""")

for name, model in MODEL_REGISTRY.items():
    status = "✅ Loaded" if model.is_loaded else "⏳ Not loaded (will load on first use)"
    st.markdown(f"- **{name}** — `{model.output_type}` — {status}")

st.markdown("### Supported Datasets")
for name, dataset_cls in DATASET_REGISTRY.items():
    ds = dataset_cls()
    st.markdown(f"- **{name}** — Emotions: {', '.join(ds.emotion_labels)}")

st.markdown("""
---

### Quick Start

1. **Download a dataset** (RAVDESS or ESD) to a directory on this machine
2. Go to **📊 Benchmark** and run a model against that dataset
3. Results are saved automatically — use **🔬 Comparison** to subtract baselines

### Dataset Downloads

| Dataset | Link | Notes |
|---------|------|-------|
| RAVDESS | [Zenodo](https://zenodo.org/records/1188976) | Download `Audio_Speech_Actors_01-24.zip` (215 MB) |
| ESD | [GitHub](https://github.com/HLTSingapore/Emotional-Speech-Data) / [Kaggle](https://www.kaggle.com/datasets/nguyenthanhlim/emotional-speech-dataset-esd) | 5 emotions, 20 speakers |
""")

# GPU info
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        st.sidebar.success(f"🖥️ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        st.sidebar.warning("⚠️ No GPU detected — inference will be slow")
except ImportError:
    st.sidebar.warning("⚠️ PyTorch not installed")
