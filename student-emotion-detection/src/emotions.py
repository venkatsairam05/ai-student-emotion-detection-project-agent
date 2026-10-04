"""Emotion label semantics: valence, arousal, and engagement weighting.

The CNN predicts one of seven FER2013 classes. Turning those into an
"engagement" signal is a small, explicit mapping rather than a learned layer,
so it can be inspected and tuned without retraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

from src.config import EMOTION_LABELS

#: Label -> (valence, arousal) on a -1..1 / 0..1 scale.
#: Valence: negative = distressed, positive = pleasant.
#: Arousal: 0 = calm, 1 = highly activated.
AFFECT_PRIMES: Dict[str, Tuple[float, float]] = {
    "angry": (-0.75, 0.85),
    "disgust": (-0.70, 0.55),
    "fear": (-0.80, 0.80),
    "happy": (0.85, 0.65),
    "sad": (-0.70, 0.25),
    "surprise": (0.10, 0.90),
    "neutral": (0.00, 0.30),
}

#: How much each emotion contributes to classroom engagement, 0..1.
#: Weight rationale (documented so it can be argued with):
#:   happy    - clearest sign of comprehension / enjoyment.
#:   surprise - alertness; engaged if the teacher just introduced something new.
#:   neutral  - baseline "present and paying attention", not disengaged.
#:   sad/fear - moderate signal: often confusion or worry, but still attentive.
#:   disgust  - mild: could be dislike of the material.
#:   angry    - lowest: frustration is the strongest disengagement cue.
ENGAGEMENT_WEIGHTS: Dict[str, float] = {
    "happy": 1.00,
    "surprise": 0.70,
    "neutral": 0.55,
    "sad": 0.35,
    "fear": 0.30,
    "disgust": 0.25,
    "angry": 0.05,
}

#: Emotions that suggest the student needs attention from the teacher.
DISTRESS_LABELS: Tuple[str, ...] = ("angry", "fear", "disgust", "sad")


@dataclass(frozen=True)
class EmotionDefinition:
    """Static metadata about one emotion class."""

    index: int
    name: str
    valence: float
    arousal: float
    engagement_weight: float
    emoji: str

    @property
    def is_distress(self) -> bool:
        """True when this emotion is considered a potential support signal."""

        return self.name in DISTRESS_LABELS


EMOTION_EMOJI: Dict[str, str] = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "sad": "😢",
    "surprise": "😮",
    "neutral": "😐",
}


def _build_definitions(labels: Sequence[str] = EMOTION_LABELS) -> Tuple[EmotionDefinition, ...]:
    """Create the ordered :class:`EmotionDefinition` tuple for ``labels``."""

    return tuple(
        EmotionDefinition(
            index=i,
            name=label,
            valence=AFFECT_PRIMES.get(label, (0.0, 0.3))[0],
            arousal=AFFECT_PRIMES.get(label, (0.0, 0.3))[1],
            engagement_weight=ENGAGEMENT_WEIGHTS.get(label, 0.5),
            emoji=EMOTION_EMOJI.get(label, "🙂"),
        )
        for i, label in enumerate(labels)
    )


EMOTIONS: Tuple[EmotionDefinition, ...] = _build_definitions()
EMOTION_BY_NAME: Mapping[str, EmotionDefinition] = {e.name: e for e in EMOTIONS}


def label_to_index(label: str, labels: Sequence[str] = EMOTION_LABELS) -> int:
    """Convert an emotion name to its class index.

    Raises ``ValueError`` for unknown labels so typos surface early instead of
    silently mapping to a wrong class.
    """

    try:
        return list(labels).index(label.lower().strip())
    except ValueError as exc:
        raise ValueError(
            f"Unknown emotion {label!r}; expected one of {list(labels)}"
        ) from exc


def index_to_label(index: int, labels: Sequence[str] = EMOTION_LABELS) -> str:
    """Convert a class index to its emotion name."""

    if not 0 <= index < len(labels):
        raise IndexError(f"Emotion index {index} out of range for {len(labels)} classes")
    return labels[index]


def engagement_score(probabilities: Mapping[str, float]) -> float:
    """Weighted engagement in ``[0, 1]`` from a label->probability mapping.

    Probabilities that do not sum to 1 are renormalized first, so callers may
    pass raw softmax outputs or already-normalized scores.
    """

    total = float(sum(probabilities.values()))
    if total <= 0.0:
        return 0.0
    score = sum(
        ENGAGEMENT_WEIGHTS.get(label, 0.5) * value for label, value in probabilities.items()
    )
    return max(0.0, min(1.0, score / total))


def valence_score(probabilities: Mapping[str, float]) -> float:
    """Expected valence in ``[-1, 1]`` from a label->probability mapping."""

    total = float(sum(probabilities.values()))
    if total <= 0.0:
        return 0.0
    weighted = sum(
        AFFECT_PRIMES.get(label, (0.0, 0.3))[0] * value
        for label, value in probabilities.items()
    )
    return max(-1.0, min(1.0, weighted / total))


def engagement_band(score: float) -> str:
    """Bucket an engagement score into a human-readable band.

    ``score`` is expected in ``[0, 1]``.
    """

    if score >= 0.70:
        return "High"
    if score >= 0.50:
        return "Moderate"
    if score >= 0.30:
        return "Low"
    return "Very Low"


def describe(label: str) -> str:
    """One-line ``emoji Name`` description for UI display."""

    definition = EMOTION_BY_NAME.get(label.lower())
    if definition is None:
        return label
    return f"{definition.emoji} {definition.name.title()}"


def to_fer2013_pixel_order() -> Tuple[str, ...]:
    """FER2013's canonical label order for reference in docs and reports."""

    return ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
