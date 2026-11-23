"""Read multiview datasets through a common observation contract."""

from .loader import AVAILABLE_DATASETS, load_dataset
from .reader import Camera, Observation, Sequence, open_dataset

__all__ = ["AVAILABLE_DATASETS", "Camera", "Observation", "Sequence", "load_dataset", "open_dataset"]
