"""Classroom analytics: aggregate faces over time into engagement signals.

A single frame tells you what one moment looks like. What a teacher actually
wants is: is the class slipping, when did it slip, and who needs a check-in. That
lives here.

:class:`SessionTracker` consumes :class:`~src.detector.FrameResult` objects and
maintains running state: per-student histories, a classroom engagement timeline,
and distress alerts with cooldowns so one struggling student does not spam the
log every frame.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from src.config import REPORTS_DIR, ensure_dirs
from src.emotions import DISTRESS_LABELS, engagement_band

try:  # pragma: no cover - import guard
    import numpy as np

    NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    NUMPY_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #


@dataclass
class Alert:
    """One classroom alert: a sustained pattern worth surfacing."""

    kind: str
    message: str
    severity: str
    timestamp: float
    faces_affected: int = 0
    share: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "faces_affected": self.faces_affected,
            "share": self.share,
        }


@dataclass
class AlertConfig:
    """Thresholds controlling when an alert fires.

    ``min_share`` is the fraction of visible faces that must show a distressed
    emotion, and ``cooldown_seconds`` suppresses repeats so the teacher sees one
    alert per event rather than one per frame.
    """

    distress_share: float = 0.4
    low_engagement: float = 0.35
    low_engagement_share: float = 0.6
    cooldown_seconds: float = 8.0
    min_faces: int = 2

    @staticmethod
    def default() -> "AlertConfig":
        return AlertConfig()


# --------------------------------------------------------------------------- #
# Per-student history
# --------------------------------------------------------------------------- #


@dataclass
class StudentTrack:
    """Rolling record for one detected face, identified by stable track id."""

    track_id: int
    labels: Deque[Tuple[float, str, float]] = field(default_factory=lambda: deque(maxlen=120))
    first_seen: float = 0.0
    last_seen: float = 0.0

    @property
    def observations(self) -> int:
        """Number of frames this track contributed to."""

        return len(self.labels)

    def label_counts(self) -> Dict[str, int]:
        """How often each emotion was predicted for this track."""

        return dict(Counter(label for _, label, _ in self.labels))

    def mean_confidence(self) -> float:
        """Average prediction confidence across observations."""

        values = [c for _, _, c in self.labels]
        return sum(values) / len(values) if values else 0.0

    def mean_engagement(self) -> float:
        """Average engagement over observations."""

        values = [e for _, _, e in self.labels]
        return sum(values) / len(values) if values else 0.0

    def dominant_label(self) -> str:
        """Most frequently predicted emotion; ties break alphabetically."""

        counts = self.label_counts()
        if not counts:
            return "neutral"
        return max(sorted(counts.items()), key=lambda kv: kv[1])[0]

    def distress_share(self) -> float:
        """Fraction of observations that were a distress emotion."""

        if not self.labels:
            return 0.0
        distressed = sum(1 for _, label, _ in self.labels if label in DISTRESS_LABELS)
        return distressed / len(self.labels)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "observations": self.observations,
            "dominant_label": self.dominant_label(),
            "mean_confidence": self.mean_confidence(),
            "mean_engagement": self.mean_engagement(),
            "distress_share": self.distress_share(),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "label_counts": self.label_counts(),
        }


# --------------------------------------------------------------------------- #
# Session tracking
# --------------------------------------------------------------------------- #


class SessionTracker:
    """Consumes frame results and maintains classroom-level aggregates.

    Usage::

        tracker = SessionTracker()
        for result in frames:
            snapshot = tracker.update(result, timestamp)
        report = tracker.report()
    """

    def __init__(
        self,
        labels: Optional[Sequence[str]] = None,
        alert_config: Optional[AlertConfig] = None,
        history_size: int = 600,
        smooth_window: int = 5,
    ) -> None:
        self.labels: List[str] = list(labels or [])
        self.alert_config = alert_config or AlertConfig.default()
        self.history_size = int(history_size)
        self.smooth_window = max(1, int(smooth_window))
        self.engagement_series: Deque[Tuple[float, float]] = deque(maxlen=self.history_size)
        self.label_totals: Counter = Counter()
        self.frame_labels: Deque[Tuple[float, Dict[str, int]]] = deque(maxlen=self.history_size)
        self.alerts: List[Alert] = []
        self.tracks: Dict[int, StudentTrack] = {}
        self._last_alert_time: Dict[str, float] = {}
        self._next_track_id = 1
        self._last_centroids: List[Tuple[int, int, int]] = []
        self._last_faces: List[Any] = []
        self.frames_seen = 0
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None

    # -- ingestion -------------------------------------------------------- #

    def update(
        self,
        result: Any,
        timestamp: float = 0.0,
    ) -> "ClassroomSnapshot":
        """Fold one :class:`~src.detector.FrameResult` into the session state."""

        self.frames_seen += 1
        if self.started_at is None:
            self.started_at = timestamp
        self.ended_at = timestamp

        engagement = float(getattr(result, "mean_engagement", 0.0))
        histogram = getattr(result, "label_histogram", dict)()
        self.engagement_series.append((timestamp, engagement))
        self.frame_labels.append((timestamp, dict(histogram)))
        self.label_totals.update(histogram)

        self._update_tracks(result, timestamp)
        self._evaluate_alerts(result, timestamp)

        return self.snapshot(timestamp)

    def _update_tracks(self, result: Any, timestamp: float) -> None:
        """Assign each face in the frame to a persistent student track."""

        faces = list(getattr(result, "faces", []))
        self._last_faces = faces
        centroids: List[Tuple[int, int, int]] = []

        for face in faces:
            x, y, w, h = face.region.as_tuple()
            centroid = (x + w // 2, y + h // 2, w)
            centroids.append(centroid)
            track_id = self._match_track(centroid)
            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
                self.tracks[track_id] = StudentTrack(track_id=track_id, first_seen=timestamp)
            track = self.tracks[track_id]
            track.last_seen = timestamp
            track.labels.append((timestamp, face.label, face.engagement))

        self._last_centroids = centroids

    def _match_track(self, centroid: Tuple[int, int, int], tolerance: float = 0.5) -> Optional[int]:
        """Find the nearest existing track by centre distance and size ratio."""

        if not self.tracks or not self._last_centroids:
            return None
        cx, cy, w = centroid
        best_id: Optional[int] = None
        best_distance = float("inf")
        for track_id, (px, py, pw) in zip(self.tracks.keys(), self._last_centroids):
            if pw <= 0:
                continue
            size_ratio = min(pw, w) / max(pw, w)
            if size_ratio < (1.0 - tolerance):
                continue
            distance = math.hypot(cx - px, cy - py)
            if distance < best_distance:
                best_distance = distance
                best_id = track_id
        return best_id

    def _evaluate_alerts(self, result: Any, timestamp: float) -> None:
        """Raise cooldown-gated alerts for distress and low engagement."""

        faces = list(getattr(result, "faces", []))
        if len(faces) < self.alert_config.min_faces:
            return

        distressed = [f for f in faces if f.label in DISTRESS_LABELS]
        share = len(distressed) / len(faces)
        if share >= self.alert_config.distress_share:
            self._raise(
                "distress_cluster",
                f"{len(distressed)} of {len(faces)} students look distressed "
                f"({share:.0%}) - check in soon.",
                "high" if share >= 0.6 else "medium",
                timestamp,
                len(distressed),
                share,
            )

        low = [f for f in faces if f.engagement < self.alert_config.low_engagement]
        low_share = len(low) / len(faces)
        if low_share >= self.alert_config.low_engagement_share:
            self._raise(
                "low_engagement",
                f"Engagement is low ({getattr(result, 'mean_engagement', 0.0):.0%}) "
                f"across {len(low)} of {len(faces)} students.",
                "medium",
                timestamp,
                len(low),
                low_share,
            )

    def _raise(
        self,
        kind: str,
        message: str,
        severity: str,
        timestamp: float,
        affected: int,
        share: float,
    ) -> None:
        """Record an alert unless it fired within the cooldown window."""

        last = self._last_alert_time.get(kind)
        if last is not None and timestamp - last < self.alert_config.cooldown_seconds:
            return
        self._last_alert_time[kind] = timestamp
        self.alerts.append(
            Alert(
                kind=kind,
                message=message,
                severity=severity,
                timestamp=timestamp,
                faces_affected=affected,
                share=share,
            )
        )

    # -- queries ---------------------------------------------------------- #

    def smoothed_engagement(self) -> List[Tuple[float, float]]:
        """Engagement series smoothed with a centered moving average."""

        values = [value for _, value in self.engagement_series]
        if not values:
            return []
        window = self.smooth_window
        if window <= 1 or len(values) < window:
            return list(self.engagement_series)
        half = window // 2
        smoothed: List[Tuple[float, float]] = []
        for index, (timestamp, _) in enumerate(self.engagement_series):
            low = max(0, index - half)
            high = min(len(values), index + half + 1)
            chunk = values[low:high]
            smoothed.append((timestamp, sum(chunk) / len(chunk)))
        return smoothed

    def snapshot(self, timestamp: float = 0.0) -> "ClassroomSnapshot":
        """Current aggregate view of the session."""

        return ClassroomSnapshot(
            timestamp=timestamp,
            mean_engagement=self.mean_engagement(),
            engagement_band=engagement_band(self.mean_engagement()),
            face_count=int(self._last_face_count()),
            label_distribution=self.label_distribution(),
            student_count=len(self.tracks),
            alert_count=len(self.alerts),
            frames_seen=self.frames_seen,
        )

    def mean_engagement(self) -> float:
        """Average engagement across the whole session."""

        values = [value for _, value in self.engagement_series]
        return sum(values) / len(values) if values else 0.0

    def _last_face_count(self) -> int:
        """Face count from the most recent frame."""

        return len(self._last_faces)

    def label_distribution(self) -> Dict[str, float]:
        """Normalized emotion shares over the session."""

        total = sum(self.label_totals.values())
        if total <= 0:
            return {}
        return {label: count / total for label, count in self.label_totals.items()}

    def trend(self, window: int = 10) -> float:
        """Change in mean engagement between the last ``window`` and prior frames.

        Positive means the class is warming up; negative means it is drifting.
        """

        values = [value for _, value in self.engagement_series]
        if len(values) < window * 2:
            return 0.0
        recent = values[-window:]
        previous = values[-2 * window : -window]
        if not previous:
            return 0.0
        return (sum(recent) / len(recent)) - (sum(previous) / len(previous))

    def report(self) -> Dict[str, Any]:
        """Full session report, JSON-serializable."""

        duration = 0.0
        if self.started_at is not None and self.ended_at is not None:
            duration = max(0.0, self.ended_at - self.started_at)

        return {
            "frames_seen": self.frames_seen,
            "duration_seconds": round(duration, 2),
            "students_detected": len(self.tracks),
            "mean_engagement": self.mean_engagement(),
            "engagement_band": engagement_band(self.mean_engagement()),
            "trend": self.trend(),
            "label_distribution": self.label_distribution(),
            "label_totals": dict(self.label_totals),
            "alerts": [a.to_dict() for a in self.alerts],
            "students": [self.tracks[k].to_dict() for k in sorted(self.tracks)],
            "engagement_series": [
                {"timestamp": round(t, 3), "engagement": round(v, 4)}
                for t, v in self.engagement_series
            ],
        }

    def reset(self) -> None:
        """Clear all accumulated state for a new session."""

        self.engagement_series.clear()
        self.frame_labels.clear()
        self.label_totals.clear()
        self.alerts.clear()
        self._last_alert_time.clear()
        self.tracks.clear()
        self._last_centroids.clear()
        self.frames_seen = 0
        self.started_at = None
        self.ended_at = None
        self._next_track_id = 1


@dataclass
class ClassroomSnapshot:
    """Point-in-time classroom aggregate returned by every :meth:`update`."""

    timestamp: float
    mean_engagement: float
    engagement_band: str
    face_count: int
    label_distribution: Dict[str, float]
    student_count: int
    alert_count: int
    frames_seen: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "mean_engagement": self.mean_engagement,
            "engagement_band": self.engagement_band,
            "face_count": self.face_count,
            "label_distribution": dict(self.label_distribution),
            "student_count": self.student_count,
            "alert_count": self.alert_count,
            "frames_seen": self.frames_seen,
        }


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #


def export_session_csv(tracker: SessionTracker, path: Optional[Path] = None) -> str:
    """Write the per-student summary to CSV; returns the file path."""

    ensure_dirs()
    path = Path(path) if path else REPORTS_DIR / "session_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [tracker.tracks[k].to_dict() for k in sorted(tracker.tracks)]
    fieldnames = [
        "track_id",
        "observations",
        "dominant_label",
        "mean_confidence",
        "mean_engagement",
        "distress_share",
        "first_seen",
        "last_seen",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(path)


def export_session_json(tracker: SessionTracker, path: Optional[Path] = None) -> str:
    """Write the full session report to JSON; returns the file path."""

    ensure_dirs()
    path = Path(path) if path else REPORTS_DIR / "session_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tracker.report(), indent=2, default=str), encoding="utf-8")
    return str(path)


def export_frame_csv(results: Iterable[Any], path: Optional[Path] = None) -> str:
    """Write one row per analysed image/frame; returns the file path.

    Accepts :class:`~src.detector.FrameResult` objects or plain dicts with the
    same keys, so it also works on batch-analysis output.
    """

    ensure_dirs()
    path = Path(path) if path else REPORTS_DIR / "frame_analysis.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source",
        "face_count",
        "mean_engagement",
        "engagement_band",
        "dominant_label",
        "dominant_share",
        "distress_count",
        "uncertain_faces",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, item in enumerate(results):
            data = item if isinstance(item, dict) else item.to_dict()
            source = data.get("source", f"item_{index}")
            writer.writerow(
                {
                    "source": source,
                    "face_count": data.get("face_count", 0),
                    "mean_engagement": round(float(data.get("mean_engagement", 0.0)), 4),
                    "engagement_band": data.get("engagement_band", ""),
                    "dominant_label": data.get("dominant_label", ""),
                    "dominant_share": round(float(data.get("dominant_share", 0.0)), 4),
                    "distress_count": data.get("distress_count", 0),
                    "uncertain_faces": sum(
                        1 for f in data.get("faces", []) if not f.get("certain", True)
                    ),
                }
            )
    return str(path)


def build_student_table(tracker: SessionTracker) -> List[Dict[str, Any]]:
    """Per-student rows shaped for ``st.dataframe``."""

    return [
        {
            "Student": f"#{track_id}",
            "Dominant emotion": str(track.dominant_label()),
            "Mean confidence": round(track.mean_confidence(), 3),
            "Engagement": round(track.mean_engagement(), 3),
            "Distress share": round(track.distress_share(), 3),
            "Frames seen": track.observations,
        }
        for track_id, track in sorted(tracker.tracks.items())
    ]


def build_engagement_dataframe(tracker: SessionTracker) -> List[Dict[str, float]]:
    """Engagement timeline rows shaped for ``st.dataframe``."""

    return [
        {"Time (s)": round(t, 2), "Engagement": round(v, 4)}
        for t, v in tracker.engagement_series
    ]


def heatmap_matrix(
    tracker: SessionTracker, labels: Sequence[str], buckets: int = 12
) -> List[List[float]]:
    """Emotion shares over time, bucketed, as stacked-bar data.

    Rows are time buckets ordered oldest first; columns follow ``labels``. Each
    cell is the fraction of faces showing that emotion inside its bucket, so
    every row sums to 1. This is the "how did the class mood shift" view the
    dashboard renders.
    """

    frames = list(tracker.frame_labels)
    if not frames or not labels:
        return []

    size = max(1, min(buckets, len(frames)))
    chunk = math.ceil(len(frames) / size)

    matrix: List[List[float]] = []
    for start in range(0, len(frames), chunk):
        window = frames[start : start + chunk]
        counts: Counter = Counter()
        for _, histogram in window:
            counts.update(histogram)
        total = sum(counts.values())
        if total <= 0:
            matrix.append([0.0] * len(labels))
            continue
        matrix.append([counts.get(label, 0) / total for label in labels])
    return matrix


def heatmap_frame(
    matrix: List[List[float]], labels: Sequence[str]
) -> List[Dict[str, object]]:
    """Reshape :func:`heatmap_matrix` output into ``st.bar_chart`` rows."""

    return [
        {"bucket": index, **{label: round(row[i], 4) for i, label in enumerate(labels)}}
        for index, row in enumerate(matrix)
        if len(row) == len(labels)
    ]
