"""Pydantic models representing movies / TV shows."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Movie(BaseModel):
    """A single movie or TV show recommendation candidate."""

    id: int | str
    title: str
    overview: str = ""
    content_type: str = Field(default="movie", description="movie or tv")
    genres: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    cast: list[str] = Field(default_factory=list)
    director: str = ""
    rating: float = 0.0
    popularity: float = 0.0
    release_date: str = ""
    release_year: Optional[int] = None
    runtime: int = 0  # minutes, 0 when unknown
    language: str = "en"
    original_language: str = "en"
    poster_path: str = ""
    backdrop_path: str = ""
    trailer_key: str = ""
    tmdb_id: Optional[int] = None
    combined_text: str = ""

    # --- Ranking output fields ------------------------------------------
    similarity_score: float = 0.0
    preference_match_score: float = 0.0
    rating_score: float = 0.0
    popularity_score: float = 0.0
    freshness_score: float = 0.0
    final_score: float = 0.0
    match_reasons: list[str] = Field(default_factory=list)

    @property
    def display_year(self) -> str:
        if self.release_year:
            return str(self.release_year)
        if self.release_date:
            return self.release_date[:4]
        return "N/A"

    @property
    def display_runtime(self) -> str:
        if not self.runtime:
            return "N/A"
        hours, minutes = divmod(int(self.runtime), 60)
        if hours:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    def to_candidate_dict(self) -> dict:
        """Serialize the fields the LLM is allowed to see/explain."""
        return {
            "title": self.title,
            "content_type": self.content_type,
            "rating": self.rating,
            "genres": self.genres,
            "overview": self.overview[:400],
            "release_year": self.release_year,
            "runtime_minutes": self.runtime,
            "language": self.language,
            "director": self.director,
            "similarity_score": round(self.similarity_score, 3),
            "match_reasons": self.match_reasons,
        }
