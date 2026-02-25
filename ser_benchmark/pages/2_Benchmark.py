"""Benchmark page — run a model on a dataset and view the confusion matrix."""

import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on path for imports
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from ui_common import show_gpu_sidebar

st.set_page_config(page_title="Benchmark", page_icon="📊", layout="wide")
show_gpu_sidebar()
st.title("Benchmark")
st.markdown("Run a model against a benchmark dataset and analyze the results.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def plot_confusion_matrix(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    diagonal_pairs: list[tuple[int, int]],
    title: str = "Confusion Matrix",
    normalized: bool = False,
    cmap: str = "Blues",
):
    """Plot a (possibly non-square) confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(max(10, len(col_labels) * 1.2), max(8, len(row_labels) * 0.9)))

    fmt = ".2f" if normalized else "d"
    data = matrix.astype(float) if normalized else matrix

    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        xticklabels=col_labels,
        yticklabels=row_labels,
        cmap=cmap,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )

    # Highlight the semantic diagonal cells
    for r, c in diagonal_pairs:
        ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor="red", linewidth=2))

    ax.set_xlabel("Predicted Label (Model Output)")
    ax.set_ylabel("True Label (Dataset)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_dimensional_by_emotion(
    results: list[dict],
    true_labels: list[str],
    unique_emotions: list[str],
):
    """Plot dimensional model output grouped by ground-truth emotion."""
    # Organize data
    data_rows = []
    for result, label in zip(results, true_labels):
        dims = result["dimensions"]
        data_rows.append({
            "emotion": label,
            "arousal": dims["arousal"],
            "dominance": dims["dominance"],
            "valence": dims["valence"],
        })

    df = pd.DataFrame(data_rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    dim_names = ["arousal", "dominance", "valence"]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]

    for ax, dim, color in zip(axes, dim_names, colors):
        # Box plot
        emotion_data = [df[df["emotion"] == e][dim].values for e in unique_emotions]
        bp = ax.boxplot(emotion_data, labels=unique_emotions, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(dim.capitalize())
        ax.set_ylabel("Value")
        ax.set_ylim(-0.1, 1.1)
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Dimensional Output Distribution by Ground-Truth Emotion", fontsize=14)
    plt.tight_layout()
    return fig


def save_benchmark_result(
    model_name: str,
    dataset_name: str,
    matrix: np.ndarray | None,
    row_labels: list[str],
    col_labels: list[str],
    diagonal_pairs: list[tuple[int, int]],
    metrics: dict | None,
    output_type: str,
    dimensional_results: list[dict] | None = None,
    true_labels: list[str] | None = None,
) -> str:
    """Save benchmark results to disk as JSON. Returns the filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{dataset_name}_{timestamp}.json"
    filepath = RESULTS_DIR / filename

    payload = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "output_type": output_type,
        "timestamp": timestamp,
        "row_labels": row_labels,
        "col_labels": col_labels,
        "diagonal_pairs": diagonal_pairs,
    }

    if matrix is not None:
        payload["matrix"] = matrix.tolist()
    if metrics is not None:
        payload["metrics"] = metrics
    if dimensional_results is not None:
        payload["dimensional_results"] = dimensional_results
    if true_labels is not None:
        payload["true_labels"] = true_labels

    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)

    return filename


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

if "model_registry" not in st.session_state:
    st.error("Model registry not initialized. Please run from the main app.py entry point.")
    st.stop()

if "dataset_registry" not in st.session_state:
    st.error("Dataset registry not initialized. Please run from the main app.py entry point.")
    st.stop()

model_registry = st.session_state["model_registry"]
dataset_registry = st.session_state["dataset_registry"]

# --- Configuration ---
col1, col2 = st.columns(2)

with col1:
    model_name = st.selectbox("Select Model", list(model_registry.keys()))
    model = model_registry[model_name]
    st.caption(f"Output type: **{model.output_type}** | Labels: {', '.join(model.labels)}")

with col2:
    dataset_name = st.selectbox("Select Dataset", list(dataset_registry.keys()))

# Dataset path — auto-detect common locations
_common_paths = {
    "RAVDESS": ["/workspace/data/RAVDESS", "/workspace/data/ravdess", "/data/RAVDESS"],
    "ESD": ["/workspace/data/ESD", "/workspace/data/esd", "/data/ESD"],
}
_default_dir = ""
for _candidate in _common_paths.get(dataset_name, []):
    if Path(_candidate).is_dir():
        _default_dir = _candidate
        break

data_dir = st.text_input(
    "Dataset directory path",
    value=_default_dir,
    placeholder="/workspace/data/RAVDESS" if dataset_name == "RAVDESS" else "/workspace/data/ESD",
    help="Absolute path to the dataset root directory on this machine.",
)

# Dataset-specific options
if dataset_name == "ESD":
    esd_col1, esd_col2 = st.columns(2)
    with esd_col1:
        esd_split = st.selectbox("Split", ["test", "evaluation", "train", "all"], index=0,
                                  help="If dataset has no split folders, all files are loaded regardless.")
    with esd_col2:
        esd_language = st.selectbox("Language", ["english", "chinese", "all"], index=0)

# Sample limit
st.markdown("---")
sample_col1, sample_col2 = st.columns(2)
with sample_col1:
    max_samples = st.number_input(
        "Max samples (0 = all)",
        min_value=0, max_value=100000, value=0, step=10,
        help="Limit the number of samples to run. Useful for quick tests. 0 = use all samples.",
    )
with sample_col2:
    stratified = st.checkbox(
        "Evenly distributed across emotions",
        value=True,
        help="Sample equally from each emotion class. Ensures balanced evaluation.",
    )

# --- Run Benchmark ---
if st.button("🚀 Run Benchmark", type="primary", disabled=not data_dir):
    if not Path(data_dir).exists():
        st.error(f"Directory not found: {data_dir}")
        st.stop()

    # Load dataset
    with st.spinner(f"Loading {dataset_name} dataset..."):
        dataset_cls = dataset_registry[dataset_name]
        dataset = dataset_cls()

        if dataset_name == "ESD":
            dataset.load(data_dir, split=esd_split, language=esd_language)
        else:
            dataset.load(data_dir)

    if len(dataset) == 0:
        st.error("No samples found in the dataset. Check your directory path and structure.")
        st.stop()

    # Subsample if requested
    import random
    all_samples = list(dataset)

    if max_samples > 0 and max_samples < len(all_samples):
        if stratified:
            # Sample equally from each emotion
            from collections import defaultdict
            by_emotion = defaultdict(list)
            for audio_path, label in all_samples:
                by_emotion[label].append((audio_path, label))

            n_emotions = len(by_emotion)
            per_emotion = max(1, max_samples // n_emotions)
            sampled = []
            for label in sorted(by_emotion.keys()):
                pool = by_emotion[label]
                random.shuffle(pool)
                sampled.extend(pool[:per_emotion])
            random.shuffle(sampled)
            all_samples = sampled
        else:
            random.shuffle(all_samples)
            all_samples = all_samples[:max_samples]

    st.info(
        f"Found **{len(dataset)}** total samples across "
        f"**{len(dataset.emotion_labels)}** emotions: {', '.join(dataset.emotion_labels)}"
        + (f"  \n**Using {len(all_samples)} samples**" + (" (stratified)" if stratified else "") if max_samples > 0 else "")
    )

    # Load model
    with st.spinner(f"Loading {model_name}..."):
        model.ensure_loaded()

    # Run inference
    true_labels = []
    pred_labels = []
    all_results = []
    progress = st.progress(0, text="Running inference...")
    start_time = time.time()
    total = len(all_samples)

    for i, (audio_path, emotion_label) in enumerate(all_samples):
        try:
            result = model.predict(audio_path)
        except Exception as e:
            st.warning(f"Skipped {audio_path}: {e}")
            continue

        true_labels.append(emotion_label)
        all_results.append(result)

        if model.output_type == "probabilities":
            pred_labels.append(result["top_label"])

        progress.progress((i + 1) / total, text=f"Running inference... {i + 1}/{total}")

    elapsed = time.time() - start_time
    progress.empty()
    st.success(f"Inference complete! {len(true_labels)} samples in {elapsed:.1f}s ({len(true_labels)/elapsed:.1f} samples/sec)")

    # --- Display Results ---
    if model.output_type == "probabilities":
        from evaluation.confusion import build_confusion_matrix, normalize_rows
        from evaluation.metrics import compute_metrics

        # Build confusion matrix
        matrix, row_labels, col_labels, diagonal_pairs = build_confusion_matrix(
            true_labels,
            pred_labels,
            true_label_order=dataset.emotion_labels,
            pred_label_order=model.labels,
        )

        # Metrics
        metrics = compute_metrics(matrix, row_labels, col_labels, diagonal_pairs)

        # Display metrics
        st.header("Metrics")
        met_cols = st.columns(4)
        met_cols[0].metric("Overall Accuracy", f"{metrics['overall_accuracy']:.1%}")
        met_cols[1].metric("Macro Precision", f"{metrics['macro_precision']:.1%}")
        met_cols[2].metric("Macro Recall", f"{metrics['macro_recall']:.1%}")
        met_cols[3].metric("Macro F1", f"{metrics['macro_f1']:.1%}")

        # Per-class table
        if metrics["per_class"]:
            st.subheader("Per-Class Metrics")
            pc_df = pd.DataFrame(metrics["per_class"])
            st.dataframe(pc_df, use_container_width=True)

        # Confusion matrices
        st.header("Confusion Matrix")

        tab_raw, tab_norm = st.tabs(["Raw Counts", "Row-Normalized"])

        with tab_raw:
            fig = plot_confusion_matrix(
                matrix, row_labels, col_labels, diagonal_pairs,
                title=f"{model_name} on {dataset_name} (Raw Counts)",
            )
            st.pyplot(fig)
            plt.close(fig)

        with tab_norm:
            norm_matrix = normalize_rows(matrix)
            fig = plot_confusion_matrix(
                norm_matrix, row_labels, col_labels, diagonal_pairs,
                title=f"{model_name} on {dataset_name} (Row-Normalized)",
                normalized=True,
                cmap="YlOrRd",
            )
            st.pyplot(fig)
            plt.close(fig)

        # Save results
        filename = save_benchmark_result(
            model_name, dataset_name, matrix, row_labels, col_labels,
            diagonal_pairs, metrics, model.output_type,
        )
        st.info(f"💾 Results saved to `results/{filename}` (use in Comparison page)")

    elif model.output_type == "dimensional":
        st.header("Dimensional Analysis")
        st.markdown(
            "This model outputs continuous values (arousal, dominance, valence) "
            "rather than discrete emotions. Confusion matrices don't apply — "
            "instead, we show how the model's dimensional output distributes "
            "across ground-truth emotions."
        )

        fig = plot_dimensional_by_emotion(all_results, true_labels, dataset.emotion_labels)
        st.pyplot(fig)
        plt.close(fig)

        # Summary statistics table
        st.subheader("Summary Statistics")
        rows = []
        for emotion in dataset.emotion_labels:
            emotion_results = [
                r["dimensions"] for r, t in zip(all_results, true_labels) if t == emotion
            ]
            if not emotion_results:
                continue
            arousals = [r["arousal"] for r in emotion_results]
            dominances = [r["dominance"] for r in emotion_results]
            valences = [r["valence"] for r in emotion_results]
            rows.append({
                "Emotion": emotion,
                "Count": len(emotion_results),
                "Arousal (mean±std)": f"{np.mean(arousals):.3f} ± {np.std(arousals):.3f}",
                "Dominance (mean±std)": f"{np.mean(dominances):.3f} ± {np.std(dominances):.3f}",
                "Valence (mean±std)": f"{np.mean(valences):.3f} ± {np.std(valences):.3f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Save results
        filename = save_benchmark_result(
            model_name, dataset_name, None, dataset.emotion_labels, model.labels,
            [], None, model.output_type,
            dimensional_results=[r for r in all_results],
            true_labels=true_labels,
        )
        st.info(f"💾 Results saved to `results/{filename}`")

    # Raw results expander
    with st.expander("🔧 Raw results (first 20 samples)"):
        for i, (result, true_label) in enumerate(zip(all_results[:20], true_labels[:20])):
            st.markdown(f"**Sample {i+1}** — True: `{true_label}`")
            st.json(result)
