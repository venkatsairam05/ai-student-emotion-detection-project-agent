"""Shared pytest fixtures.

No fixture downloads a model, reads FER2013, or touches the network. Models are
tiny randomly-initialised networks built from :mod:`src.model`, which exercises
the real code paths at a fraction of the cost.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch", reason="PyTorch is required for the model tests")
pytest.importorskip("numpy", reason="NumPy is required")

from src.config import EMOTION_LABELS, IMAGE_SIZE  # noqa: E402
from src.detector import FacePrediction, FaceRegion, FrameResult  # noqa: E402


@pytest.fixture(scope="session")
def labels() -> List[str]:
    """The canonical seven emotion labels."""

    return list(EMOTION_LABELS)


@pytest.fixture(scope="session")
def image_size() -> int:
    """Square input size used across tests."""

    return IMAGE_SIZE


@pytest.fixture
def tiny_model(labels: List[str]):
    """A randomly initialised small CNN (fast, deterministic enough to run)."""

    from src.model import TinyEmotionCNN

    torch.manual_seed(0)
    model = TinyEmotionCNN(num_classes=len(labels))
    model.eval()
    return model


@pytest.fixture
def batch_tensor(image_size: int):
    """A ``(4, 3, size, size)`` batch of zeros."""

    return torch.zeros(4, 3, image_size, image_size)


@pytest.fixture
def fake_frame():
    """A deterministic 240x320 BGR frame with two bright rectangles."""

    import numpy as np

    frame = np.full((240, 320, 3), 40, dtype=np.uint8)
    frame[40:140, 40:140] = (200, 180, 160)
    frame[60:160, 170:270] = (190, 170, 150)
    return frame


@pytest.fixture
def make_face_prediction():
    """Factory for :class:`FacePrediction` instances."""

    def _factory(
        label: str = "happy",
        engagement: float = 0.8,
        confidence: float = 0.9,
        box: tuple = (0, 0, 50, 50),
        certain: bool = True,
    ) -> FacePrediction:
        from src.emotions import EMOTION_EMOJI, engagement_band

        probabilities = {label: confidence}
        return FacePrediction(
            region=FaceRegion(*box),
            label=label,
            emoji=EMOTION_EMOJI.get(label, "🙂"),
            confidence=confidence,
            engagement=engagement,
            band=engagement_band(engagement),
            probabilities=probabilities,
            certain=certain,
        )

    return _factory


@pytest.fixture
def make_frame_result(make_face_prediction):
    """Factory for :class:`FrameResult` with sensible aggregates."""

    def _factory(faces: List[FacePrediction], fallback: bool = False) -> FrameResult:
        from src.emotions import engagement_band

        if not faces:
            return FrameResult(faces=[], fallback_used=fallback, detector="fake")
        histogram: dict = {}
        for face in faces:
            histogram[face.label] = histogram.get(face.label, 0) + 1
        dominant = max(sorted(histogram.items()), key=lambda kv: kv[1])[0]
        mean = sum(f.engagement for f in faces) / len(faces)
        return FrameResult(
            faces=faces,
            frame_count=1,
            mean_engagement=mean,
            dominant_label=dominant,
            dominant_share=histogram[dominant] / len(faces),
            distress_count=sum(1 for f in faces if f.label in {"angry", "fear", "disgust", "sad"}),
            detector="fake",
            fallback_used=fallback,
        )

    return _factory


@pytest.fixture
def make_tracker():
    """Factory for :class:`SessionTracker`."""

    def _factory(labels: List[str], **kwargs):
        from src.analytics import SessionTracker

        return SessionTracker(labels=labels, **kwargs)

    return _factory


@pytest.fixture
def image_folder(tmp_path):
    """Create a tiny image-folder dataset and return its root.

    Two classes x three 48x48 grayscale PNGs, named after real emotion labels so
    ``discover_labels`` orders them canonically.
    """

    pytest.importorskip("PIL", reason="Pillow is required for dataset tests")
    from PIL import Image

    root = tmp_path / "dataset" / "train"
    for label in ("happy", "sad"):
        directory = root / label
        directory.mkdir(parents=True)
        for index in range(3):
            Image.new("L", (48, 48), (index * 40) % 255).save(directory / f"{index}.png")

    val_root = tmp_path / "dataset" / "val"
    for label in ("happy", "sad"):
        directory = val_root / label
        directory.mkdir(parents=True)
        Image.new("L", (48, 48), 120).save(directory / "0.png")

    return tmp_path / "dataset"
