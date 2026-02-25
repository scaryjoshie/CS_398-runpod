# SER Benchmarking Suite — Implementation Plan

## Overview

A Streamlit-based benchmarking suite for Speech Emotion Recognition models, designed to run on RunPod with SSH port forwarding. Supports models with fundamentally different output types (discrete probabilities vs. continuous dimensional values) while preserving their raw output exactly as-is.

## Models (3 initial)

| Model | ID | Output Type | Labels/Dims |
|-------|----|-------------|-------------|
| emotion2vec+ Large | `iic/emotion2vec_plus_large` (via FunASR) | 9-class softmax probabilities | angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown |
| emotion2vec+ Base | `iic/emotion2vec_plus_base` (via FunASR) | 9-class softmax probabilities | (same 9) |
| Audeering Wav2Vec2 Dim | `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | Continuous regression (3 values) | arousal, dominance, valence (each ~0.0–1.0) |

**Key constraint**: The Audeering model outputs continuous dimensional values, NOT discrete emotions. It *cannot* directly produce a confusion matrix. The UI must handle this gracefully — showing dimensional distributions instead of a confusion matrix for this model.

## Datasets (2 initial)

| Dataset | Emotions | Structure |
|---------|----------|-----------|
| RAVDESS | 8: neutral, calm, happy, sad, angry, fearful, disgust, surprised | Filename-encoded (`03-01-XX-...`), 48kHz WAV, from Zenodo |
| ESD (English speakers only) | 5: neutral, happy, angry, sad, surprise | `{speaker_id}/{Emotion}/{split}/` folders, WAV, from Kaggle |

## Project Structure

```
ser_benchmark/
├── app.py                          # Streamlit entry point (model loading, page routing)
├── requirements.txt
├── models/
│   ├── __init__.py                 # Registry: MODEL_REGISTRY dict
│   ├── base.py                     # BaseModelWrapper ABC
│   ├── emotion2vec_wrapper.py      # Wraps both Large and Base (same interface)
│   └── audeering_wrapper.py        # Custom EmotionModel class + wrapper
├── datasets/
│   ├── __init__.py                 # Registry: DATASET_REGISTRY dict
│   ├── base.py                     # BaseDataset ABC
│   ├── ravdess.py                  # RAVDESS loader (filename parsing)
│   └── esd.py                      # ESD loader (folder traversal)
├── evaluation/
│   ├── __init__.py
│   ├── confusion.py                # Non-square confusion matrix builder
│   └── metrics.py                  # Accuracy, per-class precision/recall
└── pages/
    ├── 1_Single_Inference.py       # Upload audio → raw model output
    ├── 2_Benchmark.py              # Run model on dataset → confusion matrix / dim plots
    └── 3_Comparison.py             # Baseline subtraction between two runs
```

---

## Step-by-Step Implementation

### Step 1: Base classes and model wrappers

**`models/base.py`** — Abstract base:
```python
class BaseModelWrapper(ABC):
    name: str                           # Human-readable name
    output_type: str                    # "probabilities" | "dimensional"
    labels: list[str]                   # emotion names OR dimension names

    def load(self) -> None: ...         # Load model to GPU (lazy, called once)
    def predict(self, audio_path) -> dict: ...
    def predict_batch(self, audio_paths) -> list[dict]: ...
```

Output dict format varies by `output_type` — this is the whole point. No normalization:
- **probabilities**: `{"labels": ["angry", ...], "scores": [0.01, ...]}` — raw from model
- **dimensional**: `{"dimensions": {"arousal": 0.54, "dominance": 0.61, "valence": 0.40}}`

**`models/emotion2vec_wrapper.py`**:
- Single class `Emotion2VecWrapper(BaseModelWrapper)` parameterized by model_id
- Uses `funasr.AutoModel` — `model.generate(wav, granularity="utterance")`
- Strips bilingual labels to English (e.g., `"生气/angry"` → `"angry"`)
- Instantiated twice in registry: once for Large, once for Base

**`models/audeering_wrapper.py`**:
- Contains the custom `RegressionHead` and `EmotionModel` classes (from model card)
- `AudeeringWrapper(BaseModelWrapper)` with `output_type="dimensional"`
- Uses `soundfile` to load audio, `Wav2Vec2Processor` for preprocessing

### Step 2: Dataset loaders

**`datasets/base.py`**:
```python
class BaseDataset(ABC):
    name: str
    emotion_labels: list[str]           # The ground-truth label set in this dataset

    def load(self, data_dir: str) -> None: ...
    def __iter__(self) -> Iterator[tuple[str, str]]: ...  # yields (audio_path, emotion_label)
    def __len__(self) -> int: ...
```

**`datasets/ravdess.py`**:
- Parses filename 3rd position for emotion code (01–08)
- Maps: `{1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}`
- Filters to audio-only speech files (modality=03, channel=01)

**`datasets/esd.py`**:
- Walks `{data_dir}/{speaker_id}/{Emotion}/{split}/*.wav`
- Speaker filter param (default: English speakers 0011–0020)
- Split filter param (default: "test" for benchmarking, to avoid train data)
- Labels from folder names, lowercased

### Step 3: Evaluation engine

**`evaluation/confusion.py`**:
- `build_confusion_matrix(true_labels, pred_labels, true_label_order, pred_label_order)`
- Returns a numpy matrix + ordered label lists for both axes
- Smart diagonal alignment: finds common emotions between the two label sets and places those first on both axes so the diagonal represents correct predictions where possible. Remaining labels appended after.
- Row-normalization function for baseline subtraction
- Only works for models with `output_type="probabilities"` (top-1 prediction used)

**`evaluation/metrics.py`**:
- `compute_metrics(confusion_matrix, true_labels, pred_labels)` → dict with:
  - Overall accuracy (diagonal sum / total)
  - Per-class precision, recall, F1
  - Weighted averages

### Step 4: Streamlit pages

**Page 1 — Single Inference** (`pages/1_Single_Inference.py`):
- File uploader (`.wav`, `.mp3`, `.flac`)
- Model selector dropdown
- Audio playback widget
- Output display adapts to `output_type`:
  - **probabilities**: Horizontal bar chart of all class scores + top prediction highlighted
  - **dimensional**: Three gauge/bar indicators for arousal, dominance, valence
- Raw JSON output in an expander for inspection

**Page 2 — Benchmark** (`pages/2_Benchmark.py`):
- Model selector + dataset selector + dataset path input
- "Run Benchmark" button with progress bar
- For **probability models**:
  - Non-square confusion matrix heatmap (seaborn, `coolwarm` cmap)
  - Diagonal-aligned so matching emotions line up
  - Metrics table (accuracy, per-class precision/recall/F1)
  - Option to save results as JSON for later comparison
- For **dimensional models** (Audeering):
  - No confusion matrix (not applicable)
  - Instead: box plots / violin plots of arousal, dominance, valence grouped by ground-truth emotion
  - Shows whether the model's dimensional output meaningfully separates emotions
  - Mean ± std table per emotion per dimension
- Results cached to disk as JSON (matrix + metadata) for use in Comparison page

**Page 3 — Comparison / Baseline Subtraction** (`pages/3_Comparison.py`):
- Load two saved benchmark results (file selectors)
- Both must be from probability-type models to do matrix subtraction
- Row-normalize both matrices → subtract → display difference heatmap
- Color interpretation guide:
  - Diagonal near 0 = steering matches human baseline
  - Diagonal negative = underperforming vs. humans
  - Off-diagonal positive = introducing new confusions
- Side-by-side view: baseline matrix | generated matrix | difference matrix

### Step 5: App entry point and wiring

**`app.py`**:
- Streamlit multipage app setup
- Model registry with lazy loading (load on first use, cache in `st.session_state`)
- Sidebar: loaded model indicator, GPU memory info

**`requirements.txt`**:
```
streamlit>=1.30
torch>=2.0
transformers>=4.30
funasr
modelscope
soundfile
librosa
numpy
pandas
seaborn
matplotlib
scikit-learn
```

---

## Non-square confusion matrix — diagonal alignment algorithm

Since models and datasets have different label sets, the matrix won't always be square. The alignment works like this:

1. Find the intersection of ground-truth labels and prediction labels (case-insensitive fuzzy match on emotion name)
2. Order those shared labels first on both axes (same order) — these form the top-left square block with a meaningful diagonal
3. Append dataset-only labels (rows with no corresponding prediction column) below
4. Append model-only labels (columns with no corresponding truth row) to the right

Example: RAVDESS (8 emotions) × emotion2vec (9 emotions):
- Shared: angry, disgusted→disgust, fearful, happy, neutral, sad, surprised (7 match)
- RAVDESS-only row: calm
- emotion2vec-only columns: other, unknown

---

## What this plan does NOT do

- No label normalization or mapping between models — raw output preserved exactly
- No attempt to force the Audeering dimensional model into a confusion matrix
- No automatic dataset downloading (user provides the path; we document how to get them)
- No model fine-tuning — inference only
