"""Face detection + emotion inference on frames.

Two-stage pipeline:

1. **Face detection** - OpenCV Haar cascade (bundled with ``opencv-python``, so
   no extra download). When OpenCV or the cascade is unavailable, the detector
   degrades to a single whole-frame "face" region rather than failing.
2. **Emotion classification** - the trained CNN over each detected crop,
   expanded by ``FACE_BOX_PADDING`` pixels so forehead and chin survive the
   48x48 resize.

The module works on ``numpy`` arrays in BGR (OpenCV convention) and returns
plain dicts, so it is testable without a webcam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config import (
    DETECTION_CONFIDENCE,
    DETECTION_MIN_SIZE,
    FACE_BOX_PADDING,
    IMAGE_SIZE,
    MAX_FACES,
    MIN_CONFIDENCE,
    SMOOTHING_FACTOR,
    ensure_dirs,
    find_haarcascade_path,
)
from src.emotions import engagement_band, engagement_score

try:  # pragma: no cover - import guard
    import numpy as np

    NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

try:  # pragma: no cover - import guard
    import cv2

    CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False

try:  # pragma: no cover - import guard
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class FaceRegion:
    """A detected face: bounding box in pixels plus detection confidence."""

    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """``(x, y, w, h)`` for OpenCV's rectangle drawing."""

        return (self.x, self.y, self.width, self.height)

    def area(self) -> int:
        """Bounding-box area in pixels squared."""

        return max(0, self.width) * max(0, self.height)


@dataclass
class FacePrediction:
    """Emotion result for one detected face."""

    region: FaceRegion
    label: str
    emoji: str
    confidence: float
    engagement: float
    band: str
    probabilities: Dict[str, float]
    certain: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box": self.region.as_tuple(),
            "confidence_box": self.region.confidence,
            "label": self.label,
            "emoji": self.emoji,
            "confidence": self.confidence,
            "engagement": self.engagement,
            "band": self.band,
            "certain": self.certain,
            "probabilities": dict(self.probabilities),
        }


@dataclass
class FrameResult:
    """Everything the UI needs about a single analyzed frame."""

    faces: List[FacePrediction] = field(default_factory=list)
    frame_count: int = 0
    mean_engagement: float = 0.0
    dominant_label: str = "neutral"
    dominant_share: float = 0.0
    distress_count: int = 0
    detector: str = "none"
    fallback_used: bool = False

    @property
    def face_count(self) -> int:
        """Number of faces in this frame."""

        return len(self.faces)

    @property
    def engagement_band(self) -> str:
        """Classroom-level engagement band for this frame."""

        return engagement_band(self.mean_engagement)

    def label_histogram(self) -> Dict[str, int]:
        """Face counts per emotion label, ordered by the label list."""

        histogram: Dict[str, int] = {}
        for face in self.faces:
            histogram[face.label] = histogram.get(face.label, 0) + 1
        return histogram

    def mean_probabilities(self) -> Dict[str, float]:
        """Average class distribution across all faces in the frame."""

        if not self.faces:
            return {}
        labels = list(self.faces[0].probabilities.keys())
        return {
            label: sum(f.probabilities.get(label, 0.0) for f in self.faces) / len(self.faces)
            for label in labels
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "face_count": self.face_count,
            "mean_engagement": self.mean_engagement,
            "engagement_band": self.engagement_band,
            "dominant_label": self.dominant_label,
            "dominant_share": self.dominant_share,
            "distress_count": self.distress_count,
            "detector": self.detector,
            "fallback_used": self.fallback_used,
            "frame_count": self.frame_count,
            "faces": [f.to_dict() for f in self.faces],
            "histogram": self.label_histogram(),
        }


# --------------------------------------------------------------------------- #
# Face detection
# --------------------------------------------------------------------------- #


#: Parsing the cascade XML costs ~0.3s, so one instance is shared process-wide.
_CASCADE_CACHE: Dict[str, Any] = {}


def _load_cascade(path: str) -> Optional[Any]:
    """Load and memoize a Haar cascade; ``None`` when unusable."""

    if path in _CASCADE_CACHE:
        return _CASCADE_CACHE[path]
    try:
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            raise ValueError(f"empty cascade XML: {path}")
    except Exception:
        _CASCADE_CACHE[path] = None
        return None
    _CASCADE_CACHE[path] = cascade
    return cascade


class FaceDetector:
    """Haar-cascade frontal face detector with graceful degradation.

    Parameters
    ----------
    min_confidence:
        Minimum detection score. Haar ``detectMultiScale3`` reports reject
        levels rather than probabilities, so weights are rescaled by
        ``level / 5`` into ``[0, 1]`` before the threshold is applied.
    max_faces:
        Upper bound on returned regions, largest first.
    """

    def __init__(
        self,
        min_confidence: float = DETECTION_CONFIDENCE,
        min_size: int = DETECTION_MIN_SIZE,
        max_faces: int = MAX_FACES,
        cascade_path: Optional[Any] = None,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self.min_size = int(min_size)
        self.max_faces = int(max_faces)
        self.cascade_path = cascade_path
        self.cascade = None
        self.available = False
        self.reason = ""

        if not CV2_AVAILABLE:
            self.reason = "opencv-python is not installed"
            return

        path = self.cascade_path or find_haarcascade_path()
        cascade = _load_cascade(str(path)) if str(path) else None
        if cascade is None:
            self.reason = f"cascade unavailable: {path}"
            return
        self.cascade = cascade
        self.cascade_path = path
        self.available = True

    @property
    def name(self) -> str:
        """Human-readable detector name for the UI status line."""

        return "haar_frontalface" if self.available else "whole_frame_fallback"

    def detect(self, frame: "np.ndarray") -> Tuple[List[FaceRegion], bool]:
        """Detect faces in a BGR frame.

        Returns ``(regions, fallback_used)``. On fallback, a single region
        covering the whole frame is returned so the rest of the pipeline is
        unchanged.
        """

        if not NUMPY_AVAILABLE or frame is None:
            return [], True

        height, width = frame.shape[:2]
        if not self.available or self.cascade is None:
            return [FaceRegion(0, 0, int(width), int(height), confidence=0.0)], True

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            raw = self.cascade.detectMultiScale3(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_size, self.min_size),
                outputRejectLevels=True,
            )
            boxes = raw[0] if isinstance(raw, tuple) else raw
            weights = None
            if isinstance(raw, tuple) and len(raw) > 1 and raw[1] is not None:
                weights = raw[1].ravel()
        except Exception:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                boxes = self.cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(self.min_size, self.min_size)
                )
                weights = None
            except Exception:
                return [FaceRegion(0, 0, int(width), int(height), confidence=0.0)], True

        regions: List[FaceRegion] = []
        for i, (x, y, w, h) in enumerate(boxes):
            weight = float(weights[i]) if weights is not None and i < len(weights) else 1.0
            # Haar level weights are not probabilities; rescale into [0, 1].
            confidence = max(0.0, min(1.0, weight / 5.0)) if weights is not None else 1.0
            if confidence < self.min_confidence:
                continue
            regions.append(FaceRegion(int(x), int(y), int(w), int(h), confidence))

        if not regions:
            return [FaceRegion(0, 0, int(width), int(height), confidence=0.0)], True

        regions.sort(key=lambda r: r.area(), reverse=True)
        return regions[: self.max_faces], False


def pad_region(
    region: FaceRegion,
    frame_shape: Sequence[int],
    padding: int = FACE_BOX_PADDING,
) -> FaceRegion:
    """Grow a bounding box by ``padding``, clipped to the frame bounds."""

    height, width = int(frame_shape[0]), int(frame_shape[1])
    x0 = max(0, region.x - padding)
    y0 = max(0, region.y - padding)
    x1 = min(width, region.x + region.width + padding)
    y1 = min(height, region.y + region.height + padding)
    return FaceRegion(x0, y0, max(1, x1 - x0), max(1, y1 - y0), region.confidence)


def crop_face(frame: "np.ndarray", region: FaceRegion) -> "np.ndarray":
    """Crop a region out of a BGR frame, clipped and never empty."""

    height, width = frame.shape[:2]
    x0 = max(0, min(int(region.x), width - 1))
    y0 = max(0, min(int(region.y), height - 1))
    x1 = max(x0 + 1, min(int(region.x + region.width), width))
    y1 = max(y0 + 1, min(int(region.y + region.height), height))
    return frame[y0:y1, x0:x1]


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

_EMOJI_FALLBACK = {
    "angry": "!",
    "disgust": "~",
    "fear": "^",
    "happy": "+",
    "sad": "-",
    "surprise": "o",
    "neutral": ".",
}


def draw_predictions(
    frame: "np.ndarray",
    result: FrameResult,
    show_labels: bool = True,
) -> "np.ndarray":
    """Return a copy of ``frame`` with boxes, labels, and an engagement banner.

    OpenCV cannot render emoji glyphs, so text uses ASCII markers when the
    label text itself would be non-ASCII. The Streamlit layer overlays real
    emoji separately via ``st.markdown``.
    """

    if not (NUMPY_AVAILABLE and CV2_AVAILABLE) or frame is None:
        return frame

    canvas = frame.copy()
    height, width = canvas.shape[:2]

    for face in result.faces:
        x, y, w, h = face.region.as_tuple()
        engagement = face.engagement
        if engagement >= 0.6:
            color = (80, 200, 80)      # BGR green
        elif engagement >= 0.4:
            color = (60, 190, 240)     # BGR amber
        else:
            color = (60, 80, 235)      # BGR red

        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        if show_labels:
            marker = _EMOJI_FALLBACK.get(face.label, "?")
            text = f"{marker} {face.label} {face.confidence:.0%}"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            top = max(0, y - th - baseline - 4)
            cv2.rectangle(canvas, (x, top), (x + tw + 6, top + th + baseline + 4), color, -1)
            cv2.putText(
                canvas,
                text,
                (x + 3, top + th),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    banner = (
        f"faces: {result.face_count} | engagement: {result.mean_engagement:.0%} "
        f"({result.engagement_band}) | dominant: {result.dominant_label}"
    )
    (bw, bh), baseline = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(canvas, (0, 0), (min(width, bw + 12), bh + baseline + 8), (30, 30, 30), -1)
    cv2.putText(
        canvas,
        banner,
        (6, bh + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def overlay_probability_bars(
    frame: "np.ndarray",
    probabilities: Dict[str, float],
    origin: Tuple[int, int] = (10, 10),
    width: int = 220,
    bar_height: int = 14,
) -> "np.ndarray":
    """Draw a compact horizontal bar chart of a class distribution."""

    if not (NUMPY_AVAILABLE and CV2_AVAILABLE) or frame is None:
        return frame

    canvas = frame.copy()
    x0, y0 = origin
    offset = 0
    for label, value in sorted(probabilities.items(), key=lambda kv: -kv[1]):
        y = y0 + offset
        cv2.putText(
            canvas,
            f"{label[:7]:<7}",
            (x0, y + bar_height - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            canvas,
            (x0 + 62, y),
            (x0 + 62 + int(width * max(0.0, min(1.0, value))), y + bar_height),
            (200, 200, 80),
            -1,
        )
        offset += bar_height + 4
    return canvas


# --------------------------------------------------------------------------- #
# Temporal smoothing
# --------------------------------------------------------------------------- #


class EmotionSmoother:
    """Exponential moving average over per-face label probabilities.

    Raw per-frame CNN output flickers (happy -> neutral -> happy). Smoothing
    keeps a stable label for the UI without hiding genuine transitions, because
    a real change still wins within a few frames.

    Tracks faces by IoU so a person who moves slightly keeps their history.
    """

    def __init__(
        self,
        alpha: float = SMOOTHING_FACTOR,
        min_iou: float = 0.3,
        max_age: int = 8,
    ) -> None:
        self.alpha = float(max(0.0, min(1.0, alpha)))
        self.min_iou = float(min_iou)
        self.max_age = int(max_age)
        self.tracks: List[Dict[str, Any]] = []

    def update(
        self, faces: Sequence[Tuple[FaceRegion, Dict[str, float]]]
    ) -> List[Dict[str, float]]:
        """Blend ``faces`` into the tracker and return smoothed distributions.

        Existing tracks age by one frame; any track unseen for ``max_age``
        frames is dropped, so leaving the camera does not leave ghosts behind.
        """

        for track in self.tracks:
            track["age"] += 1

        if not faces:
            self._age()
            return []

        matched: set = set()
        assignments: Dict[int, Dict[str, float]] = {}

        for index, (region, distribution) in enumerate(faces):
            best_iou = 0.0
            best_position = -1
            for position, track in enumerate(self.tracks):
                if position in matched:
                    continue
                overlap = _iou(region, track["region"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_position = position
            if best_position >= 0 and best_iou >= self.min_iou:
                track = self.tracks[best_position]
                track["region"] = region
                track["age"] = 1
                track["distribution"] = {
                    label: self.alpha * value + (1 - self.alpha) * track["distribution"].get(label, 0.0)
                    for label, value in distribution.items()
                }
                assignments[index] = track["distribution"]
                matched.add(best_position)
            else:
                self.tracks.append(
                    {"region": region, "distribution": dict(distribution), "age": 1}
                )
                assignments[index] = self.tracks[-1]["distribution"]
                matched.add(len(self.tracks) - 1)

        self._age()
        return [assignments[i] for i in range(len(faces))]

    def _age(self) -> None:
        """Drop tracks that have gone unseen for ``max_age`` frames."""

        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]

    def reset(self) -> None:
        """Clear all tracked faces (call on camera restart)."""

        self.tracks.clear()


def _iou(left: FaceRegion, right: FaceRegion) -> float:
    """Intersection-over-union of two boxes."""

    ax0, ay0 = left.x, left.y
    ax1, ay1 = left.x + left.width, left.y + left.height
    bx0, by0 = right.x, right.y
    bx1, by1 = right.x + right.width, right.y + right.height
    inter_w = max(0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0, min(ay1, by1) - max(ay0, by0))
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0
    union = left.area() + right.area() - intersection
    return intersection / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


class EmotionPredictor:
    """Runs face detection + classification over frames from any source.

    Works with OpenCV images (BGR ``ndarray``), PIL images, or raw bytes. The
    model is loaded once and cached; pass an untrained model in tests and the
    same code path runs with random weights.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        labels: Optional[Sequence[str]] = None,
        detector: Optional[FaceDetector] = None,
        device: str = "auto",
        image_size: int = IMAGE_SIZE,
        min_confidence: float = MIN_CONFIDENCE,
        smooth: bool = False,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for inference. `pip install torch`.")

        from src.runtime import configure_threads

        # A webcam frame carries a handful of faces, so single-threaded inference
        # is the fastest option; the checkpoint's batch size is unknown here so
        # assume one face per batch.
        configure_threads(1, image_size)

        self.model = model
        self.labels: List[str] = list(labels) if labels else []
        self.detector = detector or FaceDetector()
        self.image_size = int(image_size)
        self.min_confidence = float(min_confidence)
        self.smoother = EmotionSmoother() if smooth else None
        self._device = device
        self.frame_count = 0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Optional[Any] = None,
        device: str = "auto",
        **kwargs: Any,
    ) -> "EmotionPredictor":
        """Load a trained checkpoint and build a predictor from it."""

        from src.checkpoint import CHECKPOINT_PATH, load_checkpoint

        model, labels, metadata = load_checkpoint(checkpoint_path or CHECKPOINT_PATH, device=device)
        predictor = cls(model=model, labels=labels, device=device, **kwargs)
        predictor.metadata = metadata  # type: ignore[attr-defined]
        return predictor

    @property
    def is_trained(self) -> bool:
        """False when running on randomly initialised weights."""

        return bool(getattr(self, "_trained_flag", True)) and self.model is not None

    def _ensure_loaded(self) -> None:
        """Load the checkpoint on first use if none was injected."""

        if self.model is None:
            from src.checkpoint import CHECKPOINT_PATH, load_checkpoint

            try:
                model, labels, _ = load_checkpoint(CHECKPOINT_PATH, device=self._device)
            except FileNotFoundError:
                return
            self.model = model
            self.labels = list(labels)

    def _to_bgr(self, source: Any) -> Optional["np.ndarray"]:
        """Normalise any supported input to a BGR ``ndarray``."""

        if source is None or not NUMPY_AVAILABLE:
            return None
        if isinstance(source, np.ndarray):
            return source
        if isinstance(source, (bytes, bytearray)):
            if not CV2_AVAILABLE:
                return None
            buffer = np.frombuffer(bytes(source), dtype=np.uint8)
            return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        convert = getattr(source, "convert", None)  # PIL.Image
        if callable(convert):
            array = np.asarray(convert("RGB"), dtype=np.uint8)
            return array[:, :, ::-1].copy()
        return None

    def predict(self, source: Any, apply_smoothing: bool = True) -> FrameResult:
        """Analyze a single frame and return a :class:`FrameResult`."""

        self._ensure_loaded()
        frame = self._to_bgr(source)
        self.frame_count += 1

        if frame is None or self.model is None:
            return FrameResult(frame_count=self.frame_count, detector=self.detector.name, fallback_used=True)

        regions, fallback = self.detector.detect(frame)
        raw_pairs: List[Tuple[FaceRegion, Dict[str, float]]] = []
        tensors = []
        for region in regions:
            padded = pad_region(region, frame.shape)
            crop = crop_face(frame, padded)
            if crop.size == 0:
                continue
            tensors.append(self._crop_to_tensor(crop))
            raw_pairs.append((padded, {}))

        if not tensors:
            return FrameResult(frame_count=self.frame_count, detector=self.detector.name, fallback_used=fallback)

        if self.smoother is not None and apply_smoothing:
            stacked = torch.cat(tensors, dim=0)
            probabilities = self.model.predict_proba(stacked).cpu().numpy()
            index_of = {id(region): i for i, (region, _) in enumerate(raw_pairs)}
            distributions = [
                {self.labels[j]: float(probabilities[index_of[id(region)], j])
                 for j in range(min(len(self.labels), probabilities.shape[1]))}
                for region, _ in raw_pairs
            ]
            smoothed = self.smoother.update([(region, dist) for (region, _), dist in zip(raw_pairs, distributions)])
        else:
            stacked = torch.cat(tensors, dim=0)
            probabilities = self.model.predict_proba(stacked).cpu().numpy()
            smoothed = [
                {self.labels[j]: float(probabilities[i, j])
                 for j in range(min(len(self.labels), probabilities.shape[1]))}
                for i in range(probabilities.shape[0])
            ]

        faces: List[FacePrediction] = []
        for (region, _), distribution in zip(raw_pairs, smoothed):
            label, confidence = _argmax(distribution)
            score = engagement_score(distribution)
            from src.emotions import EMOTION_EMOJI

            faces.append(
                FacePrediction(
                    region=region,
                    label=label,
                    emoji=EMOTION_EMOJI.get(label, "🙂"),
                    confidence=confidence,
                    engagement=score,
                    band=engagement_band(score),
                    probabilities=distribution,
                    certain=confidence >= self.min_confidence,
                )
            )

        dominant_label, dominant_share = _dominant(faces)
        mean_engagement = (
            sum(f.engagement for f in faces) / len(faces) if faces else 0.0
        )
        return FrameResult(
            faces=faces,
            frame_count=self.frame_count,
            mean_engagement=mean_engagement,
            dominant_label=dominant_label,
            dominant_share=dominant_share,
            distress_count=sum(1 for f in faces if f.label in {"angry", "fear", "disgust", "sad"}),
            detector=self.detector.name,
            fallback_used=fallback,
        )

    def _crop_to_tensor(self, crop: "np.ndarray") -> "torch.Tensor":
        """Convert a BGR crop to the model's normalized input tensor."""

        from src.evaluation import preprocess_pil

        if CV2_AVAILABLE:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        else:  # pragma: no cover - only without opencv
            rgb = crop[:, :, ::-1]
        from PIL import Image

        return preprocess_pil(Image.fromarray(rgb), self.image_size)

    def annotate(self, source: Any, result: FrameResult, show_labels: bool = True) -> "np.ndarray":
        """Draw ``result`` onto a copy of ``source``; returns a BGR array."""

        frame = self._to_bgr(source)
        if frame is None:
            return frame  # type: ignore[return-value]
        return draw_predictions(frame, result, show_labels=show_labels)

    def reset(self) -> None:
        """Reset the temporal tracker and frame counter."""

        if self.smoother is not None:
            self.smoother.reset()
        self.frame_count = 0


def _argmax(distribution: Dict[str, float]) -> Tuple[str, float]:
    """Highest-probability label and its probability; ties break alphabetically."""

    if not distribution:
        return "neutral", 0.0
    label = max(sorted(distribution.items()), key=lambda kv: kv[1])[0]
    return label, float(distribution.get(label, 0.0))


def _dominant(faces: Sequence[FacePrediction]) -> Tuple[str, float]:
    """Most common label across faces plus its share of the frame."""

    if not faces:
        return "neutral", 0.0
    counts: Dict[str, int] = {}
    for face in faces:
        counts[face.label] = counts.get(face.label, 0) + 1
    label = max(sorted(counts.items()), key=lambda kv: kv[1])[0]
    return label, counts[label] / len(faces)


def save_frame(frame: "np.ndarray", path: Any) -> str:
    """Write a BGR frame to disk as PNG; returns the path string."""

    ensure_dirs()
    if CV2_AVAILABLE and NUMPY_AVAILABLE and frame is not None:
        cv2.imwrite(str(path), frame)
    return str(path)
