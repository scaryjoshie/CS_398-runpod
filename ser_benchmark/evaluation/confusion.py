"""Confusion matrix builder with non-square support and diagonal alignment.

Handles the case where model labels and dataset labels differ:
- Shared emotions are placed first on both axes so the diagonal is meaningful
- Dataset-only labels become extra rows
- Model-only labels become extra columns
"""

import numpy as np


def _fuzzy_match(a: str, b: str) -> bool:
    """Check if two emotion labels refer to the same emotion.

    Handles minor naming differences like 'disgust' vs 'disgusted',
    'fearful' vs 'fear', 'surprised' vs 'surprise', etc.
    """
    a = a.lower().strip()
    b = b.lower().strip()

    if a == b:
        return True

    # Common SER label variations
    equivalences = [
        {"disgust", "disgusted"},
        {"fear", "fearful"},
        {"surprise", "surprised"},
        {"happy", "happiness"},
        {"sad", "sadness"},
        {"angry", "anger"},
    ]

    for group in equivalences:
        if a in group and b in group:
            return True

    return False


def align_labels(
    true_labels: list[str],
    pred_labels: list[str],
) -> tuple[list[str], list[str], list[tuple[int, int]]]:
    """Align two label sets for diagonal-meaningful confusion matrix.

    Returns:
        aligned_true: Reordered true labels (rows)
        aligned_pred: Reordered pred labels (columns)
        diagonal_pairs: List of (row_idx, col_idx) pairs that represent
                        the same emotion (i.e., the "semantic diagonal")
    """
    # Find matching pairs
    matched_true = []
    matched_pred = []
    used_pred = set()

    for t_label in true_labels:
        for p_label in pred_labels:
            if p_label not in used_pred and _fuzzy_match(t_label, p_label):
                matched_true.append(t_label)
                matched_pred.append(p_label)
                used_pred.add(p_label)
                break

    # Unmatched labels
    unmatched_true = [l for l in true_labels if l not in matched_true]
    unmatched_pred = [l for l in pred_labels if l not in used_pred]

    # Build aligned orderings: matched first, then unmatched
    aligned_true = matched_true + unmatched_true
    aligned_pred = matched_pred + unmatched_pred

    # Diagonal pairs are the first len(matched_true) indices
    diagonal_pairs = [(i, i) for i in range(len(matched_true))]

    return aligned_true, aligned_pred, diagonal_pairs


def build_confusion_matrix(
    true_labels_list: list[str],
    pred_labels_list: list[str],
    true_label_order: list[str] | None = None,
    pred_label_order: list[str] | None = None,
) -> tuple[np.ndarray, list[str], list[str], list[tuple[int, int]]]:
    """Build a (possibly non-square) confusion matrix.

    Args:
        true_labels_list: Ground truth labels for each sample.
        pred_labels_list: Predicted labels for each sample.
        true_label_order: Optional explicit ordering of true labels (rows).
        pred_label_order: Optional explicit ordering of pred labels (columns).

    Returns:
        matrix: numpy array of shape (n_true_labels, n_pred_labels)
        row_labels: Ordered true labels (rows)
        col_labels: Ordered pred labels (columns)
        diagonal_pairs: (row, col) pairs representing same-emotion matches
    """
    assert len(true_labels_list) == len(pred_labels_list), (
        f"Length mismatch: {len(true_labels_list)} true vs {len(pred_labels_list)} pred"
    )

    # Determine unique labels
    unique_true = true_label_order or sorted(set(true_labels_list))
    unique_pred = pred_label_order or sorted(set(pred_labels_list))

    # Align for meaningful diagonal
    aligned_true, aligned_pred, diagonal_pairs = align_labels(unique_true, unique_pred)

    # Build index maps
    true_idx = {label: i for i, label in enumerate(aligned_true)}
    pred_idx = {label: i for i, label in enumerate(aligned_pred)}

    # Fill the matrix
    matrix = np.zeros((len(aligned_true), len(aligned_pred)), dtype=np.int64)

    for t, p in zip(true_labels_list, pred_labels_list):
        if t in true_idx and p in pred_idx:
            matrix[true_idx[t], pred_idx[p]] += 1

    return matrix, aligned_true, aligned_pred, diagonal_pairs


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Row-normalize a confusion matrix to get probability distributions.

    Each row sums to 1.0 (or 0.0 if the row has no samples).
    Used for baseline subtraction: both matrices must be row-normalized
    before subtracting so scales match regardless of sample count.
    """
    row_sums = matrix.sum(axis=1, keepdims=True).astype(np.float64)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    return matrix.astype(np.float64) / row_sums
