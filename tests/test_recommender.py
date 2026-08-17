"""Tests for the hybrid recommender engine (semantic + metadata + ranking)."""

from __future__ import annotations

from src.models.preferences import UserPreferences


def test_recommend_returns_movies(small_recommender):
    prefs = UserPreferences()
    recs = small_recommender.recommend(prefs, query_text="sci-fi thriller")
    assert isinstance(recs, list)
    assert len(recs) > 0


def test_recommend_respects_excluded_genres(small_recommender):
    prefs = UserPreferences(excluded_genres=["Horror"])
    recs = small_recommender.recommend(prefs, query_text="movie")
    for movie in recs:
        assert "Horror" not in movie.genres


def test_recommend_respects_content_type(small_recommender):
    prefs = UserPreferences(content_type="tv")
    recs = small_recommender.recommend(prefs, query_text="show")
    for movie in recs:
        assert movie.content_type == "tv"


def test_recommend_min_rating(small_recommender):
    prefs = UserPreferences(minimum_rating=8.5)
    recs = small_recommender.recommend(prefs, query_text="thriller")
    for movie in recs:
        assert movie.rating >= 8.5


def test_recommend_top_k(small_recommender):
    prefs = UserPreferences()
    recs = small_recommender.recommend(prefs, query_text="movie", top_k=3)
    assert len(recs) <= 3


def test_similar_finds_related_titles(small_recommender):
    recs = small_recommender.similar("Inception", top_k=3)
    assert len(recs) > 0
    titles = [m.title for m in recs]
    assert "Inception" not in titles


def test_similar_unknown_title_returns_empty(small_recommender):
    recs = small_recommender.similar("This Title Does Not Exist", top_k=3)
    assert recs == []


def test_resolve_title(small_recommender):
    movie = small_recommender.resolve_title("Interstellar")
    assert movie is not None
    assert movie.title == "Interstellar"


def test_recommend_via_similar_to(small_recommender):
    prefs = UserPreferences(similar_to=["Inception"])
    recs = small_recommender.recommend(prefs)
    assert len(recs) > 0


def test_match_reasons_include_similarity(small_recommender):
    prefs = UserPreferences(similar_to=["Inception"])
    recs = small_recommender.recommend(prefs, query_text="dream heist")
    assert len(recs) > 0
    assert any(
        "Similar to Inception" in r
        for movie in recs
        for r in movie.match_reasons
    )


def test_recommender_not_ready_returns_empty():
    import pandas as pd

    from src.recommender.recommender import Recommender

    r = Recommender(index=None, metadata=None, movies=pd.DataFrame())
    assert r.ready is False
    assert r.recommend(UserPreferences()) == []
