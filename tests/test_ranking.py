"""Tests for the hybrid ranking system."""

from __future__ import annotations

from src.models.movie import Movie
from src.models.preferences import UserPreferences
from src.recommender.ranking import (
    compute_preference_match,
    freshness_score,
    rank_candidates,
    rating_score,
)


def _movie(**overrides):
    defaults = dict(
        id=1,
        title="Test Movie",
        overview="",
        genres=["Drama"],
        rating=8.0,
        popularity=50.0,
        release_year=2015,
        runtime=110,
        content_type="movie",
        language="en",
        cast=[],
        director="",
        similarity_score=0.7,
    )
    defaults.update(overrides)
    return Movie(**defaults)


def test_rating_score():
    assert rating_score(8.0, 0.0) == 0.8
    assert rating_score(6.0, 7.0) == 0.0
    assert abs(rating_score(9.5, 7.0) - 0.833333) < 1e-4
    assert rating_score(10.0, 0.0) == 1.0


def test_freshness_score():
    assert freshness_score(2025, 2025) == 1.0
    assert freshness_score(2005, 2025) == 0.0
    assert freshness_score(None, 2025) == 0.5


def test_preference_match_positive_and_negative():
    prefs = UserPreferences(genres=["Drama"], excluded_genres=["Horror"])
    good = _movie(genres=["Drama"], rating=8.5)
    bad = _movie(genres=["Horror", "Drama"])
    assert compute_preference_match(good, prefs) == 1.0
    assert compute_preference_match(bad, prefs) < 1.0


def test_preference_match_language_runtime_year():
    prefs = UserPreferences(
        languages=["English"],
        runtime_max=120,
        release_year_min=2010,
    )
    match = _movie(language="en", runtime=100, release_year=2018)
    assert compute_preference_match(match, prefs) == 1.0


def test_better_match_ranks_higher():
    prefs = UserPreferences(genres=["Comedy"], minimum_rating=7.0)
    matching = _movie(
        id=1, title="Matching Comedy", genres=["Comedy"], rating=7.5,
        similarity_score=0.8, popularity=50.0, release_year=2015,
    )
    non_matching = _movie(
        id=2, title="Non-matching Drama", genres=["Drama"], rating=6.0,
        similarity_score=0.8, popularity=50.0, release_year=2015,
    )
    ranked = rank_candidates([non_matching, matching], prefs, top_k=5)
    assert ranked[0].id == matching.id


def test_similarity_drives_order_without_prefs():
    prefs = UserPreferences()
    high = _movie(id=1, title="High", similarity_score=0.95)
    low = _movie(id=2, title="Low", similarity_score=0.3)
    ranked = rank_candidates([low, high], prefs, top_k=5)
    assert ranked[0].id == high.id


def test_top_k_respected():
    prefs = UserPreferences()
    movies = [_movie(id=i, title=f"M{i}", similarity_score=i / 10) for i in range(10)]
    ranked = rank_candidates(movies, prefs, top_k=3)
    assert len(ranked) == 3


def test_weights_are_configurable():
    prefs = UserPreferences()
    movies = [_movie(id=i, title=f"M{i}", similarity_score=i / 10) for i in range(5)]
    w = {
        "semantic_similarity": 1.0,
        "preference_match": 0.0,
        "rating": 0.0,
        "popularity": 0.0,
        "freshness": 0.0,
    }
    ranked = rank_candidates(movies, prefs, weights=w, top_k=5)
    assert ranked[0].similarity_score == max(m.similarity_score for m in movies)


def test_match_reasons_generated():
    from src.recommender.ranking import build_match_reasons

    prefs = UserPreferences(
        genres=["Drama"], minimum_rating=7.0, excluded_genres=["Horror"]
    )
    movie = _movie(genres=["Drama"], rating=8.0, similarity_score=0.8)
    reasons = build_match_reasons(movie, prefs, semantic_sim=0.8)
    assert any("Genre" in r for r in reasons)
    assert any("Rating" in r for r in reasons)
    assert any("No Horror" in r for r in reasons)
