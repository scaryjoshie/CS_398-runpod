"""Wrapper for emotion2vec+ models (Large and Base).

Uses the FunASR framework. Both models share the same interface and
9-class label set, differing only in model size and accuracy.

Labels (9): angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown
Output: softmax probabilities per class
"""

from typing import Any

from models.base import BaseModelWrapper


# The FunASR interface returns bilingual labels like "生气/angry".
# We strip to English only. The 9th label (index 8) may come as empty string.
EMOTION2VEC_LABELS = [
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
]


def _clean_label(raw_label: str) -> str:
    """Extract English portion from bilingual label like '生气/angry'."""
    if "/" in raw_label:
        return raw_label.split("/")[-1].strip()
    if raw_label.strip() == "":
        return "unknown"
    return raw_label.strip()


class Emotion2VecWrapper(BaseModelWrapper):
    """Wrapper for emotion2vec+ models via FunASR."""

    def __init__(self, model_id: str, name: str):
        self._model_id = model_id
        self._name = name
        self._model = None
        self._loaded = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_type(self) -> str:
        return "probabilities"

    @property
    def labels(self) -> list[str]:
        return EMOTION2VEC_LABELS

    def load(self) -> None:
        from funasr import AutoModel

        self._model = AutoModel(model=self._model_id, hub="hf")
        self._loaded = True

    def predict(self, audio_path: str) -> dict[str, Any]:
        self.ensure_loaded()

        res = self._model.generate(
            audio_path,
            output_dir=None,
            granularity="utterance",
            extract_embedding=False,
        )

        # res is a list of dicts, one per input
        entry = res[0]

        raw_labels = entry["labels"]
        scores = entry["scores"]

        # Clean bilingual labels to English
        labels = [_clean_label(lbl) for lbl in raw_labels]

        # Find top prediction
        top_idx = max(range(len(scores)), key=lambda i: scores[i])

        return {
            "labels": labels,
            "scores": [float(s) for s in scores],
            "top_label": labels[top_idx],
            "top_score": float(scores[top_idx]),
        }

    def predict_batch(self, audio_paths: list[str]) -> list[dict[str, Any]]:
        """Batch prediction — FunASR supports passing a list of paths."""
        self.ensure_loaded()

        results = []
        # FunASR generate can accept a list, but output ordering may vary.
        # For reliability, iterate individually.
        for path in audio_paths:
            results.append(self.predict(path))
        return results
