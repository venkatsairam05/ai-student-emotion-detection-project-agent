"""Tests for face detection, cropping, drawing, smoothing, and inference."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")

from src.detector import (
    EmotionPredictor,
    EmotionSmoother,
    FaceDetector,
    FacePrediction,
    FaceRegion,
    crop_face,
    draw_predictions,
    overlay_probability_bars,
    pad_region,
    save_frame,
)


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def test_region_area_and_tuple():
    region = FaceRegion(5, 6, 10, 20)
    assert region.as_tuple() == (5, 6, 10, 20)
    assert region.area() == 200


def test_region_area_ignores_negatives():
    assert FaceRegion(0, 0, -5, -5).area() == 0


def test_pad_region_grows_and_clips(fake_frame):
    padded = pad_region(FaceRegion(40, 40, 20, 20), fake_frame.shape, padding=10)
    assert padded.x == 30
    assert padded.width == 40


def test_pad_region_clips_at_origin(fake_frame):
    padded = pad_region(FaceRegion(0, 0, 20, 20), fake_frame.shape, padding=15)
    assert padded.x == 0
    assert padded.y == 0


def test_pad_region_clips_at_far_edge(fake_frame):
    height, width = fake_frame.shape[:2]
    padded = pad_region(FaceRegion(width - 10, height - 10, 20, 20), fake_frame.shape, padding=25)
    assert padded.x + padded.width <= width
    assert padded.y + padded.height <= height


def test_crop_face_returns_region(fake_frame):
    crop = crop_face(fake_frame, FaceRegion(40, 40, 50, 50))
    assert crop.shape[:2] == (50, 50)


def test_crop_face_clips_out_of_bounds(fake_frame):
    crop = crop_face(fake_frame, FaceRegion(300, 220, 100, 100))
    assert crop.size > 0


def test_crop_face_never_empty(fake_frame):
    assert crop_face(fake_frame, FaceRegion(-50, -50, 10, 10)).size > 0


# --------------------------------------------------------------------------- #
# FaceDetector
# --------------------------------------------------------------------------- #


def test_detector_degrades_without_opencv(monkeypatch, fake_frame):
    monkeypatch.setattr("src.detector.CV2_AVAILABLE", False)
    detector = FaceDetector()
    assert detector.available is False
    assert "opencv" in detector.reason

    regions, fallback = detector.detect(fake_frame)
    assert fallback is True
    assert len(regions) == 1
    assert regions[0].as_tuple() == (0, 0, fake_frame.shape[1], fake_frame.shape[0])


def test_detector_name_reflects_availability(monkeypatch):
    monkeypatch.setattr("src.detector.CV2_AVAILABLE", False)
    assert FaceDetector().name == "whole_frame_fallback"


def test_detector_on_none_frame(fake_frame):
    regions, fallback = FaceDetector().detect(None)
    assert regions == []
    assert fallback is True


def test_detector_respects_max_faces(fake_frame):
    detector = FaceDetector(max_faces=2)
    regions, _ = detector.detect(fake_frame)
    assert len(regions) <= 2


class _StubCascade:
    """Stand-in for ``cv2.CascadeClassifier`` returning fixed boxes and weights."""

    def __init__(self, boxes, weights):
        self.boxes = boxes
        self.weights = weights
        self.calls = 0

    def empty(self):
        return False

    def detectMultiScale3(self, *args, **kwargs):  # noqa: N802 - mirrors OpenCV
        self.calls += 1
        import numpy as np

        return (
            np.array(self.boxes, dtype=np.int32),
            np.array(self.weights, dtype=np.float32),
        )

    def detectMultiScale(self, *args, **kwargs):  # noqa: N802 - mirrors OpenCV
        import numpy as np

        return np.array(self.boxes, dtype=np.int32)


def _stubbed_detector(monkeypatch, boxes, weights, **kwargs):
    detector = FaceDetector(**kwargs)
    detector.cascade = _StubCascade(boxes, weights)
    detector.available = True
    monkeypatch.setattr(detector, "cascade", detector.cascade)
    return detector


def test_detector_parses_detectmultiScale3_output(monkeypatch, fake_frame):
    detector = _stubbed_detector(monkeypatch, [(10, 10, 40, 40), (90, 20, 30, 30)], [5.0, 3.0])
    regions, fallback = detector.detect(fake_frame)

    assert fallback is False
    assert len(regions) == 2
    assert regions[0].as_tuple() == (10, 10, 40, 40)
    assert regions[0].confidence == pytest.approx(1.0)
    assert regions[1].confidence == pytest.approx(0.6)


def test_detector_rejects_low_confidence_boxes(monkeypatch, fake_frame):
    detector = _stubbed_detector(
        monkeypatch, [(10, 10, 40, 40)], [1.0], min_confidence=0.5
    )
    regions, fallback = detector.detect(fake_frame)
    assert fallback is True
    assert regions[0].as_tuple() == (0, 0, fake_frame.shape[1], fake_frame.shape[0])


def test_detector_sorts_by_area_largest_first(monkeypatch, fake_frame):
    detector = _stubbed_detector(
        monkeypatch, [(10, 10, 20, 20), (10, 10, 60, 60)], [5.0, 5.0]
    )
    regions, _ = detector.detect(fake_frame)
    assert regions[0].width == 60
    assert regions[1].width == 20


def test_detector_truncates_to_max_faces(monkeypatch, fake_frame):
    boxes = [(i * 20, 0, 30, 30) for i in range(6)]
    detector = _stubbed_detector(monkeypatch, boxes, [5.0] * 6, max_faces=3)
    regions, _ = detector.detect(fake_frame)
    assert len(regions) == 3


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #


def test_draw_predictions_returns_copy(fake_frame, make_frame_result):
    from src.detector import CV2_AVAILABLE

    if not CV2_AVAILABLE:
        pytest.skip("opencv-python not installed")
    result = make_frame_result([])
    canvas = draw_predictions(fake_frame, result)
    assert canvas is not fake_frame
    assert canvas.shape == fake_frame.shape


def test_overlay_probability_bars(fake_frame):
    from src.detector import CV2_AVAILABLE

    if not CV2_AVAILABLE:
        pytest.skip("opencv-python not installed")
    out = overlay_probability_bars(fake_frame, {"happy": 0.8, "sad": 0.2})
    assert out.shape == fake_frame.shape


def test_save_frame_returns_path(tmp_path, fake_frame):
    path = save_frame(fake_frame, tmp_path / "out.png")
    assert path.endswith("out.png")


# --------------------------------------------------------------------------- #
# Smoothing
# --------------------------------------------------------------------------- #


def test_smoother_passes_through_first_frame():
    smoother = EmotionSmoother(alpha=0.5)
    distributions = [{"happy": 1.0}]
    result = smoother.update([(FaceRegion(0, 0, 10, 10), distributions[0])])
    assert result[0]["happy"] == pytest.approx(1.0)


def test_smoother_blends_toward_previous():
    smoother = EmotionSmoother(alpha=0.5, min_iou=0.0)
    box = FaceRegion(0, 0, 10, 10)
    smoother.update([(box, {"happy": 1.0})])
    result = smoother.update([(box, {"happy": 0.0})])
    assert result[0]["happy"] == pytest.approx(0.5)


def test_smoother_tracks_moved_face_as_same_track():
    smoother = EmotionSmoother(alpha=0.5, min_iou=0.1)
    smoother.update([(FaceRegion(0, 0, 20, 20), {"happy": 1.0})])
    assert len(smoother.tracks) == 1
    smoother.update([(FaceRegion(2, 2, 20, 20), {"happy": 1.0})])
    assert len(smoother.tracks) == 1


def test_smoother_creates_new_track_for_distant_face():
    smoother = EmotionSmoother(alpha=0.5, min_iou=0.3)
    smoother.update([(FaceRegion(0, 0, 20, 20), {"happy": 1.0})])
    smoother.update([(FaceRegion(500, 500, 20, 20), {"sad": 1.0})])
    assert len(smoother.tracks) == 2


def test_smoother_expires_old_tracks():
    smoother = EmotionSmoother(alpha=0.5, max_age=2)
    smoother.update([(FaceRegion(0, 0, 20, 20), {"happy": 1.0})])
    for _ in range(4):
        smoother.update([])
    assert len(smoother.tracks) == 0


def test_smoother_reset_clears_state():
    smoother = EmotionSmoother()
    smoother.update([(FaceRegion(0, 0, 10, 10), {"happy": 1.0})])
    smoother.reset()
    assert smoother.tracks == []


def test_smoother_empty_input_returns_empty():
    assert EmotionSmoother().update([]) == []


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #


@pytest.fixture
def predictor(tiny_model, labels):
    return EmotionPredictor(model=tiny_model, labels=labels, detector=FaceDetector(), device="cpu")


def test_predictor_returns_frame_result(predictor, fake_frame, labels):
    result = predictor.predict(fake_frame)
    assert result.frame_count == 1
    assert result.face_count >= 1
    assert all(f.label in labels for f in result.faces)


def test_predictor_accepts_bytes(predictor, tmp_path):
    from src.detector import CV2_AVAILABLE

    if not CV2_AVAILABLE:
        pytest.skip("opencv-python not installed")
    from PIL import Image

    path = tmp_path / "frame.png"
    Image.new("RGB", (120, 120), (200, 180, 160)).save(path)
    result = predictor.predict(path.read_bytes())
    assert result.frame_count == 1


def test_predictor_accepts_pil(predictor):
    from PIL import Image

    result = predictor.predict(Image.new("RGB", (80, 80), (210, 190, 170)))
    assert result.face_count >= 1


def test_predictor_handles_none(predictor):
    result = predictor.predict(None)
    assert result.face_count == 0
    assert result.fallback_used is True


def test_predictor_without_model_returns_empty(tmp_path, fake_frame):
    """With no checkpoint on disk the predictor yields a flagged empty result."""

    from src.detector import FaceDetector

    predictor = EmotionPredictor(model=None, labels=[], detector=FaceDetector(), device="cpu")
    predictor._checkpoint_path = tmp_path / "absent.pt"
    predictor._ensure_loaded = lambda: None
    predictor.model = None

    result = predictor.predict(fake_frame)
    assert result.face_count == 0
    assert result.fallback_used is True


def test_predictor_counts_frames(predictor, fake_frame):
    for expected in (1, 2, 3):
        assert predictor.predict(fake_frame).frame_count == expected
    predictor.reset()
    assert predictor.frame_count == 0


def test_predictor_flags_low_confidence(tiny_model, labels, fake_frame):
    predictor = EmotionPredictor(
        model=tiny_model, labels=labels, detector=FaceDetector(), device="cpu", min_confidence=0.99
    )
    result = predictor.predict(fake_frame)
    assert any(not face.certain for face in result.faces)


def test_frame_result_helpers(make_face_prediction, make_frame_result):
    faces = [
        make_face_prediction("happy", 0.9),
        make_face_prediction("happy", 0.8),
        make_face_prediction("sad", 0.3),
    ]
    result = make_frame_result(faces)
    histogram = result.label_histogram()
    assert histogram == {"happy": 2, "sad": 1}
    assert result.dominant_label == "happy"
    assert 0.0 <= result.mean_engagement <= 1.0
    assert result.engagement_band in {"High", "Moderate", "Low", "Very Low"}


def test_frame_result_to_dict_is_serializable(make_face_prediction, make_frame_result):
    import json

    result = make_frame_result([make_face_prediction("happy")])
    json.dumps(result.to_dict())
    assert set(result.to_dict()) >= {"face_count", "mean_engagement", "faces"}


def test_empty_frame_result_defaults(make_face_prediction):
    from src.detector import FrameResult

    result = FrameResult()
    assert result.face_count == 0
    assert result.label_histogram() == {}
    assert result.mean_probabilities() == {}
