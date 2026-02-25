"""Wrapper for audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim.

This model outputs continuous dimensional emotion values:
  - Arousal (low energy → high energy)
  - Dominance (submissive → dominant)
  - Valence (negative → positive)

Each value is approximately in [0, 1] range.

Requires custom model classes from the HuggingFace model card since the
standard pipeline does not support this architecture.

NOTE: All heavy imports (torch, transformers, numpy) are deferred to load()
to keep app startup fast.
"""

from typing import Any

from models.base import BaseModelWrapper


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
DIMENSION_NAMES = ["arousal", "dominance", "valence"]


def _build_model_classes():
    """Build the custom model classes at load time, not import time.

    This avoids importing torch/transformers on every Streamlit page load.
    """
    import torch
    import torch.nn as nn
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
    )

    class RegressionHead(nn.Module):
        """Classification head with regression output."""

        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features, **kwargs):
            x = features
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            x = self.out_proj(x)
            return x

    class EmotionModel(Wav2Vec2PreTrainedModel):
        """Wav2Vec2-based dimensional emotion model."""

        # transformers 5.x expects these class-level attributes for weight tying
        _tied_weights_keys = []
        all_tied_weights_keys = set()

        def __init__(self, config):
            super().__init__(config)
            # Ensure instance-level attribute exists (some transformers versions
            # only set it via __init_subclass__ or post_init)
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = set()
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.post_init()

        def forward(self, input_values):
            outputs = self.wav2vec2(input_values)
            hidden_states = outputs[0]
            hidden_states = torch.mean(hidden_states, dim=1)
            logits = self.classifier(hidden_states)
            return hidden_states, logits

        @classmethod
        def _can_set_experts_implementation(cls):
            # Not an MoE model — skip the sys.modules check in transformers 5.x
            return False

    return EmotionModel


class AudeeringDimWrapper(BaseModelWrapper):
    """Wrapper for the Audeering dimensional emotion model."""

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "Audeering Wav2Vec2 Dim"

    @property
    def output_type(self) -> str:
        return "dimensional"

    @property
    def labels(self) -> list[str]:
        return DIMENSION_NAMES

    def load(self) -> None:
        import sys as _sys
        import types
        import torch
        from transformers import Wav2Vec2Processor

        EmotionModel = _build_model_classes()

        # transformers 5.x does sys.modules[cls.__module__] during __init__.
        # Since this module is loaded via sys.path manipulation, it may not
        # be registered. Create a stub so the lookup doesn't KeyError.
        # (The stub has no __file__, so transformers returns False — which is
        # correct: this isn't an MoE model.)
        _mod_name = EmotionModel.__module__
        if _mod_name not in _sys.modules:
            _sys.modules[_mod_name] = types.ModuleType(_mod_name)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        self._model = EmotionModel.from_pretrained(MODEL_ID).to(self._device)
        self._model.eval()
        self._loaded = True

    def _load_audio(self, audio_path: str) -> tuple:
        """Load audio file as float32 mono 16kHz."""
        import numpy as np
        import soundfile as sf
        import librosa

        signal, sr = sf.read(audio_path)

        # Convert stereo to mono
        if signal.ndim > 1:
            signal = signal[:, 0]

        # Resample to 16kHz if needed
        if sr != 16000:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=16000)
            sr = 16000

        return signal.astype(np.float32), sr

    def predict(self, audio_path: str) -> dict[str, Any]:
        import numpy as np
        import torch

        self.ensure_loaded()

        signal, sr = self._load_audio(audio_path)

        # Process through the Wav2Vec2 processor
        inputs = self._processor(signal, sampling_rate=sr)
        input_values = inputs["input_values"][0]
        input_values = torch.from_numpy(
            np.array(input_values).reshape(1, -1)
        ).to(self._device)

        with torch.no_grad():
            _hidden_states, logits = self._model(input_values)

        values = logits.detach().cpu().numpy()[0]  # shape: (3,)

        return {
            "dimensions": {
                "arousal": float(values[0]),
                "dominance": float(values[1]),
                "valence": float(values[2]),
            }
        }
