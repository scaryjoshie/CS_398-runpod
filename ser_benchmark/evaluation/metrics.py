"""Metrics computation from confusion matrices.

Supports non-square matrices by computing metrics only for
label pairs that have a semantic match (the aligned diagonal).
"""

import numpy as np


def compute_metrics(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    diagonal_pairs: list[tuple[int, int]],
) -> dict:
    """Compute classification metrics from a confusion matrix.

    Args:
        matrix: Confusion matrix of shape (n_rows, n_cols).
        row_labels: True label names (rows).
        col_labels: Predicted label names (columns).
        diagonal_pairs: (row, col) pairs representing same-emotion matches.

    Returns:
        Dict with:
            - overall_accuracy: fraction of samples on the semantic diagonal
            - total_samples: total number of predictions
            - per_class: list of dicts with per-emotion precision, recall, f1
            - macro_precision, macro_recall, macro_f1: macro-averaged scores
    """
    total_samples = int(matrix.sum())
    if total_samples == 0:
        return {
            "overall_accuracy": 0.0,
            "total_samples": 0,
            "per_class": [],
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
        }

    # Overall accuracy: sum of diagonal pair cells / total
    correct = sum(int(matrix[r, c]) for r, c in diagonal_pairs)
    overall_accuracy = correct / total_samples

    # Per-class metrics
    per_class = []
    precisions = []
    recalls = []
    f1s = []

    for row_idx, col_idx in diagonal_pairs:
        true_label = row_labels[row_idx]
        pred_label = col_labels[col_idx]
        tp = int(matrix[row_idx, col_idx])

        # Recall: TP / (sum of this row) — how many of this true emotion were correctly predicted
        row_sum = int(matrix[row_idx, :].sum())
        recall = tp / row_sum if row_sum > 0 else 0.0

        # Precision: TP / (sum of this column) — how many predictions of this emotion were correct
        col_sum = int(matrix[:, col_idx].sum())
        precision = tp / col_sum if col_sum > 0 else 0.0

        # F1
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class.append({
            "true_label": true_label,
            "pred_label": pred_label,
            "true_positives": tp,
            "support": row_sum,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    n_classes = len(diagonal_pairs) if diagonal_pairs else 1

    return {
        "overall_accuracy": round(overall_accuracy, 4),
        "total_samples": total_samples,
        "per_class": per_class,
        "macro_precision": round(sum(precisions) / n_classes, 4),
        "macro_recall": round(sum(recalls) / n_classes, 4),
        "macro_f1": round(sum(f1s) / n_classes, 4),
    }
