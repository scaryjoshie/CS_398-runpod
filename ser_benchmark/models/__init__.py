from models.emotion2vec_wrapper import Emotion2VecWrapper
from models.audeering_wrapper import AudeeringDimWrapper

MODEL_REGISTRY = {
    "emotion2vec+ Large": Emotion2VecWrapper(
        model_id="iic/emotion2vec_plus_large",
        name="emotion2vec+ Large",
    ),
    "emotion2vec+ Base": Emotion2VecWrapper(
        model_id="iic/emotion2vec_plus_base",
        name="emotion2vec+ Base",
    ),
    "Audeering Wav2Vec2 Dim": AudeeringDimWrapper(),
}
