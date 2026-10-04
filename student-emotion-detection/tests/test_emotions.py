"""Tests for the emotion label mapping and engagement scoring."""

from __future__ import annotations

import pytest

from src.emotions import (
    DISTRESS_LABELS,
    EMOTIONS,
    ENGAGEMENT_WEIGHTS,
    engagement_band,
    engagement_score,
    index_to_label,
    label_to_index,
    valence_score,
)


def test_every_emotion_has_a_weight(labels):
    assert set(ENGAGEMENT_WEIGHTS) == set(labels)


def test_definition_indices_match_labels(labels):
    for definition in EMOTIONS:
        assert definition.index == label_to_index(definition.name)


def test_label_to_index_roundtrips(labels):
    for index, label in enumerate(labels):
        assert label_to_index(label) == index
        assert index_to_label(index) == label


def test_label_to_index_is_case_insensitive():
    assert label_to_index("HAPPY") == label_to_index("happy")


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        label_to_index("elated")


def test_index_out_of_range_raises(labels):
    with pytest.raises(IndexError):
        index_to_label(len(labels))
    with pytest.raises(IndexError):
        index_to_label(-1)


def test_engagement_score_is_confidence_weighted():
    assert engagement_score({"happy": 1.0}) == pytest.approx(1.0)
    assert engagement_score({"angry": 1.0}) == pytest.approx(
        ENGAGEMENT_WEIGHTS["angry"]
    )


def test_engagement_score_is_bounded():
    assert 0.0 <= engagement_score({"happy": 1.0, "angry": 1.0}) <= 1.0


def test_engagement_score_renormalizes_unnormalized_input():
    normalized = engagement_score({"happy": 0.5, "neutral": 0.5})
    unnormalized = engagement_score({"happy": 5.0, "neutral": 5.0})
    assert normalized == pytest.approx(unnormalized)


def test_engagement_score_handles_empty_input():
    assert engagement_score({}) == 0.0


def test_happy_beats_neutral_beats_angry():
    happy = engagement_score({"happy": 1.0})
    neutral = engagement_score({"neutral": 1.0})
    angry = engagement_score({"angry": 1.0})
    assert happy > neutral > angry


def test_valence_score_signs():
    assert valence_score({"happy": 1.0}) > 0
    assert valence_score({"sad": 1.0}) < 0
    assert valence_score({"neutral": 1.0}) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("score", "band"),
    [(0.95, "High"), (0.70, "High"), (0.60, "Moderate"), (0.50, "Moderate"),
     (0.40, "Low"), (0.30, "Low"), (0.10, "Very Low")],
)
def test_engagement_bands(score, band):
    assert engagement_band(score) == band


def test_distress_labels_are_subset_of_known_emotions(labels):
    assert set(DISTRESS_LABELS).issubset(set(labels))


def test_distress_flag_matches_label_list():
    for definition in EMOTIONS:
        assert definition.is_distress == (definition.name in DISTRESS_LABELS)
