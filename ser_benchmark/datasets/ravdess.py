"""RAVDESS dataset loader.

The Ryerson Audio-Visual Database of Emotional Speech and Song.
Download from: https://zenodo.org/records/1188976

Directory structure after extraction:
    Audio_Speech_Actors_01-24/
        Actor_01/
            03-01-01-01-01-01-01.wav
            03-01-01-01-01-02-01.wav
            ...
        Actor_02/
            ...

Filename convention (7-part numerical identifier):
    Position 1 - Modality:    01=full-AV, 02=video-only, 03=audio-only
    Position 2 - Vocal:       01=speech, 02=song
    Position 3 - Emotion:     01-08 (see EMOTION_MAP)
    Position 4 - Intensity:   01=normal, 02=strong
    Position 5 - Statement:   01="Kids are talking...", 02="Dogs are sitting..."
    Position 6 - Repetition:  01=1st, 02=2nd
    Position 7 - Actor:       01-24 (odd=male, even=female)

Audio: 16-bit 48kHz WAV
"""

import os
from pathlib import Path
from typing import Iterator

from datasets.base import BaseDataset

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

EMOTION_LABELS = list(EMOTION_MAP.values())


class RAVDESSDataset(BaseDataset):
    """Loader for the RAVDESS speech emotion dataset."""

    def __init__(self):
        self._samples: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return "RAVDESS"

    @property
    def emotion_labels(self) -> list[str]:
        return EMOTION_LABELS

    def load(
        self,
        data_dir: str,
        speech_only: bool = True,
        modality: str = "audio-only",
    ) -> None:
        """Load RAVDESS samples from disk.

        Args:
            data_dir: Path to the extracted RAVDESS directory
                      (e.g. "Audio_Speech_Actors_01-24" or parent dir).
            speech_only: If True, only load speech (vocal channel 01), not song.
            modality: Filter by modality — "audio-only" (03) is default.
        """
        modality_code = {"full-av": "01", "video-only": "02", "audio-only": "03"}
        target_modality = modality_code.get(modality, "03")

        self._samples = []
        data_path = Path(data_dir)

        # Walk all .wav files recursively
        for wav_file in sorted(data_path.rglob("*.wav")):
            parts = wav_file.stem.split("-")
            if len(parts) != 7:
                continue

            file_modality, vocal_channel, emotion_code = parts[0], parts[1], parts[2]

            # Filter
            if file_modality != target_modality:
                continue
            if speech_only and vocal_channel != "01":
                continue

            emotion = EMOTION_MAP.get(emotion_code)
            if emotion is None:
                continue

            self._samples.append((str(wav_file), emotion))

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)
