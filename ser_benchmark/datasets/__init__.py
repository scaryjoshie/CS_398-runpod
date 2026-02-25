from datasets.ravdess import RAVDESSDataset
from datasets.esd import ESDDataset

DATASET_REGISTRY = {
    "RAVDESS": RAVDESSDataset,
    "ESD": ESDDataset,
}
