from abc import ABC, abstractmethod
from typing import Iterator


class BaseDataset(ABC):
    """Abstract base for SER benchmark datasets.

    Each dataset yields (audio_path, emotion_label) pairs.
    Labels are the dataset's own native labels — no mapping applied.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable dataset name."""
        ...

    @property
    @abstractmethod
    def emotion_labels(self) -> list[str]:
        """Ordered list of emotion labels present in this dataset."""
        ...

    @abstractmethod
    def load(self, data_dir: str, **kwargs) -> None:
        """Scan the dataset directory and index all samples.

        Args:
            data_dir: Root directory where the dataset is stored.
            **kwargs: Dataset-specific options (split, speaker filter, etc.)
        """
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[str, str]]:
        """Iterate over (audio_path, emotion_label) pairs."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Total number of samples."""
        ...
