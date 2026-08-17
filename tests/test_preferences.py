"""Tests for preference extraction (rule-based + merging + Pydantic)."""

from __future__ import annotations

from src.chatbot.preference_extractor import (
    extract_preferences_rule_based,
    to_canonical_genre,
)
from src.models.preferences import UserPreferences


def test_extract_comedy_after_2020():
    prefs = extract_preferences_rule_based("I want comedy movies after 2020")
    assert "Comedy" in prefs.genres
    assert prefs.release_year_min == 2020
    assert prefs.content_type == "movie"


def test_negative_preference_horror():
    prefs = extract_preferences_rule_based("I don't want horror")
    assert "Horror" in prefs.excluded_genres
    assert "Horror" not in prefs.genres


def test_runtime_under_2_hours():
    prefs = extract_preferences_rule_based("something under 2 hours")
    assert prefs.runtime_max == 120


def test_similar_to_inception():
    prefs = extract_preferences_rule_based("movies similar to Inception")
    assert "Inception" in prefs.similar_to


def test_less_serious_mood():
    prefs = extract_preferences_rule_based(
        "something like Interstellar but less serious"
    )
    assert "Interstellar" in prefs.similar_to
    assert prefs.mood


def test_korean_dramas():
    prefs = extract_preferences_rule_based("Recommend Korean dramas similar to Squid Game")
    assert prefs.content_type == "tv"
    assert "Korean" in prefs.languages
    assert "Squid Game" in prefs.similar_to


def test_no_negatives_collide_with_genres():
    prefs = extract_preferences_rule_based(
        "I want a funny English movie after 2020 and under 2 hours. I don't want horror."
    )
    assert "Comedy" in prefs.genres
    assert "Horror" in prefs.excluded_genres
    assert "English" in prefs.languages
    assert prefs.release_year_min == 2020
    assert prefs.runtime_max == 120


def test_family_friendly():
    prefs = extract_preferences_rule_based("give me a family-friendly movie")
    assert prefs.family_friendly is True
    assert prefs.content_type == "movie"


def test_merge_dedupes_and_keeps_scalars():
    first = UserPreferences(genres=["Comedy"], content_type="movie")
    second = UserPreferences(
        genres=["Comedy", "Drama"],
        excluded_genres=["Horror"],
        release_year_min=2015,
    )
    merged = first.merge(second)
    assert merged.genres == ["Comedy", "Drama"]
    assert merged.excluded_genres == ["Horror"]
    assert merged.release_year_min == 2015
    assert merged.content_type == "movie"


def test_merge_later_scalar_wins():
    first = UserPreferences(runtime_max=120, minimum_rating=6.5)
    second = UserPreferences(runtime_max=90, minimum_rating=7.5)
    merged = first.merge(second)
    assert merged.runtime_max == 90
    assert merged.minimum_rating == 7.5


def test_is_empty():
    assert UserPreferences().is_empty() is True
    assert UserPreferences(genres=["Drama"]).is_empty() is False


def test_pydantic_validation_rejects_bad_types():
    try:
        UserPreferences(minimum_rating="not-a-number")
    except Exception:
        pass
    else:
        raise AssertionError("Expected validation to reject invalid rating")


def test_to_canonical_genre():
    assert to_canonical_genre("sci-fi") == "Science Fiction"
    assert to_canonical_genre("Comedy") == "Comedy"
    assert to_canonical_genre("nonsense") is None
