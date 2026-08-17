"""Hybrid, weighted, explainable ranking for recommendation candidates.

Pipeline:
    1. Candidate pool (semantic search results + preference-matched rows).
    2. Metadata + preference filters applied (hard filters).
    3. Normalized component scores: semantic, preference, rating, popularity,
       freshness.
    4. Weighted final score (weights configurable).
    5. Top-K selection with human-readable match reasons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from config.settings import get_settings
from src.models.movie import Movie
from src.models.preferences import UserPreferences

DEFAULT_WEIGHTS: dict[str, float] = {
    "semantic_similarity": 0.50,
    "preference_match": 0.20,
    "rating": 0.15,
    "popularity": 0.10,
    "freshness": 0.05,
}


@dataclass
class RankingResult:
    """A movie enriched with its ranking component scores."""

    movie: Movie
    semantic_similarity: float = 0.0
    preference_match: float = 0.0
    rating_score: float = 0.0
    popularity_score: float = 0.0
    freshness_score: float = 0.0
    final_score: float = 0.0
    match_reasons: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.match_reasons is None:
            self.match_reasons = []


def _normalize(values: list[float]) -> list[float]:
    """Min-max normalize a list into [0, 1] (all-equal -> zeros)."""
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    min_v, max_v = float(arr.min()), float(arr.max())
    if max_v == min_v:
        return [0.0] * len(arr)
    return ((arr - min_v) / (max_v - min_v)).tolist()


def rating_score(rating: float, min_rating: float) -> float:
    """Rating component: linear from 0 (at floor) to 1 (at 10)."""
    floor = max(0.0, min_rating)
    if rating <= floor:
        return 0.0
    return min(1.0, (rating - floor) / max(0.01, 10.0 - floor))


def freshness_score(release_year: Optional[int], reference_year: int) -> float:
    """Recency component: 1.0 for the reference year, decaying to 0 over ~20y."""
    if not release_year:
        return 0.5
    return float(max(0.0, min(1.0, 1.0 - (reference_year - release_year) / 20.0)))


def compute_preference_match(movie: Movie, prefs: UserPreferences) -> float:
    """Fraction of applicable preference constraints the movie satisfies.

    Returns a value in [0, 1] where 1.0 means "matches everything".
    """
    checks: list[bool] = []

    if prefs.genres:
        genre_hit = bool(set(movie.genres) & set(prefs.genres))
        checks.append(genre_hit)

    if prefs.excluded_genres:
        genre_conflict = bool(set(movie.genres) & set(prefs.excluded_genres))
        checks.append(not genre_conflict)

    if prefs.languages:
        langs = {lang.lower() for lang in prefs.languages}
        movie_lang = str(movie.language or "").lower()
        checks.append(
            any(
                movie_lang == lang
                or (lang == "english" and movie_lang in ("en", "en-us"))
                for lang in langs
            )
        )

    if prefs.minimum_rating:
        checks.append(movie.rating >= prefs.minimum_rating)

    if prefs.release_year_min is not None:
        year = movie.release_year or (
            int(movie.release_date[:4]) if movie.release_date else None
        )
        checks.append(year is not None and year >= prefs.release_year_min)

    if prefs.release_year_max is not None:
        year = movie.release_year or (
            int(movie.release_date[:4]) if movie.release_date else None
        )
        checks.append(year is not None and year <= prefs.release_year_max)

    if prefs.runtime_min is not None:
        checks.append(movie.runtime >= prefs.runtime_min)

    if prefs.runtime_max is not None:
        checks.append(movie.runtime > 0 and movie.runtime <= prefs.runtime_max)

    if prefs.actors:
        checks.append(bool(set(movie.cast) & set(prefs.actors)))

    if prefs.directors:
        checks.append(
            any(
                d.lower() in str(movie.director).lower()
                for d in prefs.directors
            )
        )

    if prefs.family_friendly:
        forbidden = {"Horror", "Erotic", "Mature"}
        checks.append(not (set(movie.genres) & forbidden))

    if prefs.content_type in ("movie", "tv"):
        checks.append(movie.content_type == prefs.content_type)

    if prefs.keywords:
        combined = " ".join(movie.keywords).lower()
        checks.append(any(k.lower() in combined for k in prefs.keywords))

    if not checks:
        return 0.5  # no constraints to evaluate -> neutral
    return float(sum(checks) / len(checks))


def build_match_reasons(
    movie: Movie,
    prefs: UserPreferences,
    semantic_sim: float,
    similarity_source: Optional[str] = None,
) -> list[str]:
    """Human-readable, verified reasons a candidate matched."""
    reasons: list[str] = []

    matched_genres = set(movie.genres) & set(prefs.genres)
    if matched_genres:
        reasons.append(f"Genre: {', '.join(sorted(matched_genres))}")

    if prefs.excluded_genres and not (set(movie.genres) & set(prefs.excluded_genres)):
        reasons.append(f"No {'/'.join(prefs.excluded_genres)} content")

    if prefs.minimum_rating and movie.rating >= prefs.minimum_rating:
        reasons.append(f"Rating {movie.rating:.1f} ≥ {prefs.minimum_rating}")

    if similarity_source:
        reasons.append(f"Similar to {similarity_source}")

    if semantic_sim >= 0.35:
        reasons.append(f"Strong semantic match ({semantic_sim:.2f})")

    year = movie.release_year
    if year is not None:
        if prefs.release_year_min is not None and year >= prefs.release_year_min:
            reasons.append(f"Released {year} (after {prefs.release_year_min})")
        if prefs.release_year_max is not None and year <= prefs.release_year_max:
            reasons.append(f"Released {year} (before {prefs.release_year_max})")

    if prefs.runtime_max and movie.runtime and movie.runtime <= prefs.runtime_max:
        reasons.append(f"Runtime {movie.runtime}m under {prefs.runtime_max}m")

    if prefs.runtime_min and movie.runtime and movie.runtime >= prefs.runtime_min:
        reasons.append(f"Runtime {movie.runtime}m over {prefs.runtime_min}m")

    if prefs.mood and (prefs.mood.lower() in movie.combined_text.lower()[:4000]):
        reasons.append(f"Matches mood: {prefs.mood}")

    if prefs.family_friendly and not (
        set(movie.genres) & {"Horror", "Erotic", "Mature"}
    ):
        reasons.append("Family friendly")

    matched_cast = set(movie.cast) & set(prefs.actors)
    if matched_cast:
        reasons.append(f"Starring {', '.join(sorted(matched_cast))}")

    if prefs.directors and any(
        d.lower() in str(movie.director).lower() for d in prefs.directors
    ):
        reasons.append(f"Directed by {movie.director}")

    return reasons[:8]


def rank_candidates(
    candidates: Iterable[Movie],
    preferences: UserPreferences,
    weights: Optional[dict[str, float]] = None,
    top_k: Optional[int] = None,
) -> list[Movie]:
    """Rank candidate movies by the weighted hybrid score.

    Hard-filtering happens *before* ranking via ``filter_candidates``;
    this function only scores and orders.
    """
    settings = get_settings()
    weights = weights or settings.ranking_weights
    k = top_k or settings.top_k
    cands = list(candidates)
    if not cands:
        return []

    semantics = [c.similarity_score for c in cands]
    matches = [compute_preference_match(c, preferences) for c in cands]
    ratings = [
        rating_score(c.rating, preferences.minimum_rating) for c in cands
    ]
    popularities = [float(c.popularity or 0.0) for c in cands]
    years = [c.release_year for c in cands]

    norm_sem = _normalize(semantics)
    norm_pop = _normalize(popularities)
    year_ref = max([y for y in years if y is not None] + [2025])
    freshness = [freshness_score(y, year_ref) for y in years]

    results: list[RankingResult] = []
    for i, movie in enumerate(cands):
        final = (
            weights.get("semantic_similarity", 0.0) * norm_sem[i]
            + weights.get("preference_match", 0.0) * matches[i]
            + weights.get("rating", 0.0) * ratings[i]
            + weights.get("popularity", 0.0) * norm_pop[i]
            + weights.get("freshness", 0.0) * freshness[i]
        )
        results.append(
            RankingResult(
                movie=movie,
                semantic_similarity=norm_sem[i],
                preference_match=matches[i],
                rating_score=ratings[i],
                popularity_score=norm_pop[i],
                freshness_score=freshness[i],
                final_score=final,
                match_reasons=movie.match_reasons,
            )
        )

    results.sort(key=lambda r: r.final_score, reverse=True)
    top = results[:k]

    ranked: list[Movie] = []
    for result in top:
        movie = result.movie
        movie.similarity_score = result.semantic_similarity
        movie.preference_match_score = result.preference_match
        movie.rating_score = result.rating_score
        movie.popularity_score = result.popularity_score
        movie.freshness_score = result.freshness_score
        movie.final_score = result.final_score
        movie.match_reasons = result.match_reasons or movie.match_reasons
        ranked.append(movie)
    return ranked
