"""Central configuration for the student emotion detection system.

Every tunable knob lives here so training, inference, and the Streamlit UI all
read from one source of truth. Values can be overridden with environment
variables (see :func:`env_int` / :func:`env_float` / :func:`env_bool`) which is
what the Dockerfile and CI scripts rely on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
TEST_DIR = PROCESSED_DIR / "test"

CHECKPOINT_PATH = MODEL_DIR / "emotion_cnn.pt"
LABELS_PATH = MODEL_DIR / "labels.json"
TRAIN_HISTORY_PATH = ARTIFACTS_DIR / "training_history.json"

#: Environment override, e.g. to point at a cascade file shipped with the app.
HAARCASCADE_PATH = Path(os.getenv("HAARCASCADE_PATH", ""))

FACE_CASCADE_NAME = "haarcascade_frontalface_default.xml"


def find_haarcascade_path() -> Path:
    """Resolve the Haar cascade used for face detection.

    Order of preference: the ``HAARCASCADE_PATH`` environment variable, then
    OpenCV's bundled data directory. Returns a non-existent path when OpenCV is
    not installed so callers can degrade instead of crashing.
    """

    if HAARCASCADE_PATH.name == FACE_CASCADE_NAME and HAARCASCADE_PATH.is_file():
        return HAARCASCADE_PATH
    try:
        import cv2  # type: ignore
    except Exception:
        return HAARCASCADE_PATH
    bundled = Path(cv2.data.haarcascades) / FACE_CASCADE_NAME
    if bundled.is_file():
        return bundled
    return HAARCASCADE_PATH


# --------------------------------------------------------------------------- #
# Environment helpers
# --------------------------------------------------------------------------- #


def env_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back to ``default``."""

    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back to ``default``."""

    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean from the environment (``1/true/yes/y/on`` are truthy)."""

    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# --------------------------------------------------------------------------- #
# Dataset / model
# --------------------------------------------------------------------------- #

IMAGE_SIZE = env_int("IMAGE_SIZE", 48)
NUM_CLASSES = env_int("NUM_CLASSES", 7)

#: FER2013 class order. Index -> label. Never reorder without retraining.
EMOTION_LABELS = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)

BATCH_SIZE = env_int("BATCH_SIZE", 64)
LEARNING_RATE = env_float("LEARNING_RATE", 1e-3)
WEIGHT_DECAY = env_float("WEIGHT_DECAY", 1e-4)
EPOCHS = env_int("EPOCHS", 30)
EARLY_STOPPING_PATIENCE = env_int("EARLY_STOPPING_PATIENCE", 5)
NUM_WORKERS = env_int("NUM_WORKERS", 0)
SEED = env_int("SEED", 42)
DEVICE = os.getenv("DEVICE", "auto")


@dataclass
class ModelConfig:
    """Architecture hyper-parameters for the emotion CNN."""

    num_classes: int = NUM_CLASSES
    in_channels: int = 3
    width: int = 32
    dropout: float = 0.4
    spatial_dropout: float = 0.25

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    """Everything needed to reproduce a training run."""

    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    patience: int = EARLY_STOPPING_PATIENCE
    seed: int = SEED
    num_workers: int = NUM_WORKERS
    label_smoothing: float = 0.05
    use_class_weights: bool = True
    checkpoint_path: str = str(CHECKPOINT_PATH)
    history_path: str = str(TRAIN_HISTORY_PATH)
    device: str = DEVICE
    augmentation: Dict[str, float] = field(
        default_factory=lambda: {
            "random_horizontal_flip": 0.5,
            "random_rotation_degrees": 12,
            "random_erasing_p": 0.25,
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

DETECTION_CONFIDENCE = env_float("DETECTION_CONFIDENCE", 0.25)
DETECTION_MIN_SIZE = env_int("DETECTION_MIN_SIZE", 40)
MAX_FACES = env_int("MAX_FACES", 24)
SMOOTHING_FACTOR = env_float("SMOOTHING_FACTOR", 0.6)
WEBCAM_MAX_SIDE = env_int("WEBCAM_MAX_SIDE", 640)
FACE_BOX_PADDING = env_int("FACE_BOX_PADDING", 12)

#: Probability below which a prediction is reported as "uncertain".
MIN_CONFIDENCE = env_float("MIN_CONFIDENCE", 0.35)

#: Sampling stride when walking a video file, in frames.
VIDEO_SAMPLE_STRIDE = env_int("VIDEO_SAMPLE_STRIDE", 10)


def ensure_dirs() -> None:
    """Create the directories the app writes to. Safe to call repeatedly."""

    for path in (DATA_DIR, RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR, MODEL_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
