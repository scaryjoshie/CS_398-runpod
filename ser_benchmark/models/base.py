from abc import ABC, abstractmethod
from typing import Any


class BaseModelWrapper(ABC):
    """Abstract base class for SER model wrappers.

    Each wrapper preserves the model's exact native output format.
    No label normalization or mapping is performed.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        ...

    @property
    @abstractmethod
    def output_type(self) -> str:
        """One of: 'probabilities', 'dimensional'.

        - probabilities: model outputs a score per discrete emotion class
        - dimensional: model outputs continuous values (e.g. arousal, dominance, valence)
        """
        ...

    @property
    @abstractmethod
    def labels(self) -> list[str]:
        """Label names for the model's output.

        For 'probabilities': emotion class names (e.g. ['angry', 'happy', ...])
        For 'dimensional': dimension names (e.g. ['arousal', 'dominance', 'valence'])
        """
        ...

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded into memory."""
        return getattr(self, "_loaded", False)

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory / GPU. Called once, lazily."""
        ...

    @abstractmethod
    def predict(self, audio_path: str) -> dict[str, Any]:
        """Run inference on a single audio file.

        Returns a dict whose structure depends on output_type:

        probabilities:
            {
                "labels": ["angry", "disgusted", ...],
                "scores": [0.01, 0.02, ...],
                "top_label": "neutral",
                "top_score": 0.70,
            }

        dimensional:
            {
                "dimensions": {"arousal": 0.54, "dominance": 0.61, "valence": 0.40}
            }
        """
        ...

    def predict_batch(self, audio_paths: list[str]) -> list[dict[str, Any]]:
        """Run inference on multiple audio files.

        Default implementation: iterate and call predict().
        Subclasses may override for batched inference.
        """
        return [self.predict(p) for p in audio_paths]

    def ensure_loaded(self) -> None:
        """Load model if not already loaded."""
        if not self.is_loaded:
            self.load()
