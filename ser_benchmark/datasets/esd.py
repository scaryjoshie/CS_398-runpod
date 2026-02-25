"""ESD (Emotional Speech Database) loader.

Download from: https://github.com/HLTSingapore/Emotional-Speech-Data
Also on Kaggle: https://www.kaggle.com/datasets/nguyenthanhlim/emotional-speech-dataset-esd

Directory structure:
    ESD/
        0001/                           # Speaker ID (0001-0010 Chinese, 0011-0020 English)
            Angry/
                train/
                    0001_000351.wav
                    ...
                test/
                    ...
                evaluation/
                    ...
            Happy/
            Neutral/
            Sad/
            Surprise/
            transcription.txt

5 emotions: Angry, Happy, Neutral, Sad, Surprise
Audio: WAV files
"""

import os
from pathlib import Path
from typing import Iterator

from datasets.base import BaseDataset

# ESD folder names → lowercase labels
ESD_EMOTIONS = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
ESD_LABELS = [e.lower() for e in ESD_EMOTIONS]

# Speaker ID ranges
CHINESE_SPEAKERS = [f"{i:04d}" for i in range(1, 11)]    # 0001-0010
ENGLISH_SPEAKERS = [f"{i:04d}" for i in range(11, 21)]   # 0011-0020


class ESDDataset(BaseDataset):
    """Loader for the ESD emotional speech dataset."""

    def __init__(self):
        self._samples: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return "ESD"

    @property
    def emotion_labels(self) -> list[str]:
        return ESD_LABELS

    def load(
        self,
        data_dir: str,
        split: str = "test",
        language: str = "english",
        speakers: list[str] | None = None,
    ) -> None:
        """Load ESD samples from disk.

        Args:
            data_dir: Path to the ESD root directory.
            split: Which split to use — "train", "test", or "evaluation".
            language: "english", "chinese", or "all".
            speakers: Specific speaker IDs to include (e.g. ["0011", "0012"]).
                      Overrides the language filter if provided.
        """
        self._samples = []
        data_path = Path(data_dir)

        # Determine which speakers to include
        if speakers is not None:
            target_speakers = set(speakers)
        elif language == "english":
            target_speakers = set(ENGLISH_SPEAKERS)
        elif language == "chinese":
            target_speakers = set(CHINESE_SPEAKERS)
        else:  # "all"
            target_speakers = set(CHINESE_SPEAKERS + ENGLISH_SPEAKERS)

        # Walk the directory structure
        for speaker_dir in sorted(data_path.iterdir()):
            if not speaker_dir.is_dir():
                continue

            speaker_id = speaker_dir.name
            if speaker_id not in target_speakers:
                continue

            for emotion_folder in ESD_EMOTIONS:
                emotion_dir = speaker_dir / emotion_folder / split
                if not emotion_dir.is_dir():
                    # Try lowercase folder name as fallback
                    emotion_dir = speaker_dir / emotion_folder.lower() / split
                    if not emotion_dir.is_dir():
                        continue

                emotion_label = emotion_folder.lower()

                for wav_file in sorted(emotion_dir.glob("*.wav")):
                    self._samples.append((str(wav_file), emotion_label))

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)
