"""ESD (Emotional Speech Database) loader.

Download from: https://github.com/HLTSingapore/Emotional-Speech-Data
Also on Kaggle: https://www.kaggle.com/datasets/nguyenthanhlim/emotional-speech-dataset-esd

Supports TWO directory structures:

Structure A (official / with splits):
    ESD/
        0001/
            Angry/
                train/
                    0001_000351.wav
                test/
                evaluation/
            Happy/
            ...

Structure B (Kaggle download / flat):
    ESD/
        Emotion Speech Dataset/       <-- optional extra nesting
            0011/
                Angry/
                    0011_000351.wav    <-- WAV files directly, no split folders
                Happy/
                ...

The loader auto-detects which structure is present.

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


def _find_speaker_root(data_path: Path) -> Path:
    """Find the directory that directly contains speaker folders (0001, 0011, etc.).

    Handles extra nesting like 'Emotion Speech Dataset/' from Kaggle downloads.
    """
    # Check if speaker folders are directly here
    for child in data_path.iterdir():
        if child.is_dir() and child.name.isdigit() and len(child.name) == 4:
            return data_path

    # Check one level deeper (e.g., "Emotion Speech Dataset/")
    for child in data_path.iterdir():
        if child.is_dir() and not child.name.startswith("."):
            for grandchild in child.iterdir():
                if grandchild.is_dir() and grandchild.name.isdigit() and len(grandchild.name) == 4:
                    return child

    # Fallback: just use the provided path
    return data_path


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
            split: Which split to use — "train", "test", "evaluation", or "all".
                   If the dataset has no split folders (flat structure), this is ignored
                   and all files are loaded.
            language: "english", "chinese", or "all".
            speakers: Specific speaker IDs to include (e.g. ["0011", "0012"]).
                      Overrides the language filter if provided.
        """
        self._samples = []
        data_path = Path(data_dir)

        # Auto-detect the actual root containing speaker folders
        speaker_root = _find_speaker_root(data_path)

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
        for speaker_dir in sorted(speaker_root.iterdir()):
            if not speaker_dir.is_dir():
                continue

            speaker_id = speaker_dir.name
            if speaker_id not in target_speakers:
                continue

            for emotion_folder in ESD_EMOTIONS:
                # Try exact case first, then lowercase
                emotion_base = speaker_dir / emotion_folder
                if not emotion_base.is_dir():
                    emotion_base = speaker_dir / emotion_folder.lower()
                    if not emotion_base.is_dir():
                        continue

                emotion_label = emotion_folder.lower()

                # Check if split subdirectories exist (Structure A)
                split_dir = emotion_base / split if split != "all" else None
                if split_dir and split_dir.is_dir():
                    # Structure A: has train/test/evaluation splits
                    if split == "all":
                        for sub in ["train", "test", "evaluation"]:
                            sub_dir = emotion_base / sub
                            if sub_dir.is_dir():
                                for wav_file in sorted(sub_dir.glob("*.wav")):
                                    self._samples.append((str(wav_file), emotion_label))
                    else:
                        for wav_file in sorted(split_dir.glob("*.wav")):
                            self._samples.append((str(wav_file), emotion_label))
                else:
                    # Structure B: WAV files directly in emotion folder (no splits)
                    # Check if there are WAV files here
                    wav_files = sorted(emotion_base.glob("*.wav"))
                    if wav_files:
                        for wav_file in wav_files:
                            self._samples.append((str(wav_file), emotion_label))
                    elif split == "all":
                        # Try all split subdirs as a last resort
                        for sub in ["train", "test", "evaluation"]:
                            sub_dir = emotion_base / sub
                            if sub_dir.is_dir():
                                for wav_file in sorted(sub_dir.glob("*.wav")):
                                    self._samples.append((str(wav_file), emotion_label))

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)
