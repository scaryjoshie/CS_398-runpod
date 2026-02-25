"""Comparison page — baseline subtraction between two benchmark runs.

Load two saved benchmark results (both must be probability-type models),
row-normalize both matrices, subtract, and visualize the difference.

This isolates your model's steering quality from the evaluator's inherent bias.
"""

import sys
import json
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

st.set_page_config(page_title="Comparison", page_icon="🔬", layout="wide")
st.title("🔬 Baseline Subtraction & Comparison")
st.markdown("""
Compare two benchmark runs by subtracting their row-normalized confusion matrices.
This compensates for the SER classifier's inherent inaccuracies.

**Typical use:** Run the same classifier on a ground-truth human speech dataset (baseline)
and on your generated emotion-steered audio, then subtract to isolate your steering quality.
""")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_result(filepath: str) -> dict:
    with open(filepath) as f:
        return json.load(f)


def find_result_files() -> list[str]:
    """Find all saved benchmark result JSON files."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        [f.name for f in RESULTS_DIR.glob("*.json")],
        reverse=True,  # Most recent first
    )


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    cmap: str = "Blues",
    center: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = ".2f",
):
    """Plot a heatmap from a matrix."""
    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 1.1), max(6, len(row_labels) * 0.8)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        xticklabels=col_labels,
        yticklabels=row_labels,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def align_matrices_for_subtraction(
    result_a: dict,
    result_b: dict,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Align two result matrices to have the same rows and columns.

    Only keeps rows/columns that exist in BOTH results.
    Returns row-normalized versions of both.
    """
    rows_a = set(result_a["row_labels"])
    rows_b = set(result_b["row_labels"])
    cols_a = set(result_a["col_labels"])
    cols_b = set(result_b["col_labels"])

    shared_rows = sorted(rows_a & rows_b)
    shared_cols = sorted(cols_a & cols_b)

    if not shared_rows or not shared_cols:
        raise ValueError(
            "No overlapping labels between the two results. "
            f"Result A rows: {result_a['row_labels']}, "
            f"Result B rows: {result_b['row_labels']}"
        )

    def extract_submatrix(result: dict, target_rows: list[str], target_cols: list[str]) -> np.ndarray:
        matrix = np.array(result["matrix"])
        row_idx = {label: i for i, label in enumerate(result["row_labels"])}
        col_idx = {label: i for i, label in enumerate(result["col_labels"])}

        sub = np.zeros((len(target_rows), len(target_cols)))
        for ri, row_label in enumerate(target_rows):
            for ci, col_label in enumerate(target_cols):
                if row_label in row_idx and col_label in col_idx:
                    sub[ri, ci] = matrix[row_idx[row_label], col_idx[col_label]]
        return sub

    matrix_a = extract_submatrix(result_a, shared_rows, shared_cols)
    matrix_b = extract_submatrix(result_b, shared_rows, shared_cols)

    # Row-normalize
    def row_normalize(m):
        row_sums = m.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return m / row_sums

    return row_normalize(matrix_a), row_normalize(matrix_b), shared_rows, shared_cols


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

result_files = find_result_files()

if not result_files:
    st.warning("No benchmark results found. Run a benchmark first on the Benchmark page.")
    st.stop()

# Filter to probability-type results only
prob_files = []
for f in result_files:
    try:
        data = load_result(RESULTS_DIR / f)
        if data.get("output_type") == "probabilities":
            prob_files.append(f)
    except Exception:
        continue

if len(prob_files) < 2:
    st.warning(
        f"Need at least 2 probability-type benchmark results for comparison. "
        f"Found {len(prob_files)}. Run more benchmarks first."
    )
    st.stop()

# Select two results
col1, col2 = st.columns(2)
with col1:
    st.subheader("Baseline (A)")
    st.caption("e.g., classifier run on human speech dataset")
    file_a = st.selectbox("Result file A", prob_files, key="file_a")

with col2:
    st.subheader("Generated (B)")
    st.caption("e.g., classifier run on your emotion-steered audio")
    default_b = 1 if len(prob_files) > 1 else 0
    file_b = st.selectbox("Result file B", prob_files, index=default_b, key="file_b")

if file_a == file_b:
    st.warning("Select two different result files.")
    st.stop()

# Load results
result_a = load_result(RESULTS_DIR / file_a)
result_b = load_result(RESULTS_DIR / file_b)

# Show info
info_col1, info_col2 = st.columns(2)
with info_col1:
    st.markdown(f"**Model:** {result_a['model_name']}  \n**Dataset:** {result_a['dataset_name']}  \n**Samples:** {result_a['metrics']['total_samples']}")
with info_col2:
    st.markdown(f"**Model:** {result_b['model_name']}  \n**Dataset:** {result_b['dataset_name']}  \n**Samples:** {result_b['metrics']['total_samples']}")

# --- Compute subtraction ---
if st.button("🔬 Compute Difference (B − A)", type="primary"):
    try:
        norm_a, norm_b, shared_rows, shared_cols = align_matrices_for_subtraction(result_a, result_b)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    diff = norm_b - norm_a

    st.header("Results")

    # Side by side: A | B | Diff
    tab_side, tab_diff, tab_interpret = st.tabs([
        "Side-by-Side",
        "Difference Matrix",
        "Interpretation Guide",
    ])

    with tab_side:
        c1, c2 = st.columns(2)
        with c1:
            fig = plot_heatmap(norm_a, shared_rows, shared_cols,
                             f"A: {result_a['model_name']} on {result_a['dataset_name']} (normalized)",
                             cmap="YlOrRd")
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            fig = plot_heatmap(norm_b, shared_rows, shared_cols,
                             f"B: {result_b['model_name']} on {result_b['dataset_name']} (normalized)",
                             cmap="YlOrRd")
            st.pyplot(fig)
            plt.close(fig)

    with tab_diff:
        fig = plot_heatmap(
            diff, shared_rows, shared_cols,
            f"Difference: B − A",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
        )
        st.pyplot(fig)
        plt.close(fig)

        # Diagonal summary
        st.subheader("Diagonal Summary (Same-Emotion Accuracy Difference)")
        diag_data = []
        for i, (row_l, col_l) in enumerate(zip(shared_rows, shared_cols)):
            if i < diff.shape[0] and i < diff.shape[1]:
                diag_data.append({
                    "Emotion": row_l,
                    "Baseline (A)": f"{norm_a[i, i]:.3f}",
                    "Generated (B)": f"{norm_b[i, i]:.3f}",
                    "Difference (B−A)": f"{diff[i, i]:+.3f}",
                    "Interpretation": (
                        "✅ On par" if abs(diff[i, i]) < 0.05
                        else "⬆️ Better than baseline" if diff[i, i] > 0
                        else "⬇️ Worse than baseline"
                    ),
                })
        if diag_data:
            st.dataframe(pd.DataFrame(diag_data), use_container_width=True)

    with tab_interpret:
        st.markdown("""
        ### How to Read the Difference Matrix

        The difference matrix is computed as: **C_diff = C_generated − C_baseline**

        Both matrices are **row-normalized** first (each row sums to 1.0) so that
        different sample sizes don't affect the comparison.

        #### Diagonal Values (Same Emotion)
        | Value | Meaning |
        |-------|---------|
        | **≈ 0** | Your model steers this emotion about as well as real human speech sounds to the classifier |
        | **< 0 (blue)** | Your model under-performs vs. human speech for this emotion |
        | **> 0 (red)** | Your model actually triggers this emotion *more strongly* than human speech |

        #### Off-Diagonal Values
        | Value | Meaning |
        |-------|---------|
        | **≈ 0** | Similar confusion pattern as human speech |
        | **> 0 (red)** | Your model introduces *new* confusions that humans don't make |
        | **< 0 (blue)** | Your model has *fewer* confusions than humans for this pair |

        #### Example
        If cell (sad, neutral) = +0.15, it means when you ask for "sad",
        the classifier reads it as "neutral" 15% more often than it does with real human sad speech.
        This tells you your sadness steering isn't expressive enough.
        """)
