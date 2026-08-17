"""Pydantic models describing structured user preferences."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# Canonical genre vocabulary used for normalization.
KNOWN_GENRES: set[str] = {
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
    "Romance", "Science Fiction", "Thriller", "TV Movie", "War", "Western",
}

# Canonical content types.
CONTENT_TYPES: tuple[str, ...] = ("movie", "tv", "both")


class UserPreferences(BaseModel):
    """Structured, conversation-aggregated preferences.

    Every field is optional; unknown constraints simply stay unset.
    """

    genres: list[str] = Field(default_factory=list)
    excluded_genres: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    mood: str = ""
    content_type: str = "both"
    release_year_min: Optional[int] = None
    release_year_max: Optional[int] = None
    minimum_rating: float = 0.0
    runtime_min: Optional[int] = None
    runtime_max: Optional[int] = None
    similar_to: list[str] = Field(default_factory=list)
    actors: list[str] = Field(default_factory=list)
    directors: list[str] = Field(default_factory=list)
    country: str = ""
    family_friendly: bool = False
    keywords: list[str] = Field(default_factory=list)
    sort_by: str = "relevance"

    def merge(self, other: "UserPreferences") -> "UserPreferences":
        """Merge ``other`` into this preference set (deduplicating lists).

        Later (conversation) values override scalar fields when the incoming
        value is actually populated.
        """
        merged = self.model_copy(deep=True)
        merged.genres = _merge_unique(merged.genres, other.genres)
        merged.excluded_genres = _merge_unique(
            merged.excluded_genres, other.excluded_genres
        )
        merged.languages = _merge_unique(merged.languages, other.languages)
        merged.similar_to = _merge_unique(merged.similar_to, other.similar_to)
        merged.actors = _merge_unique(merged.actors, other.actors)
        merged.directors = _merge_unique(merged.directors, other.directors)
        merged.keywords = _merge_unique(merged.keywords, other.keywords)
        merged.mood = other.mood or merged.mood
        merged.country = other.country or merged.country
        # Only a concrete content type overrides; "both" (unset) never wins.
        if other.content_type in ("movie", "tv"):
            merged.content_type = other.content_type
        if other.release_year_min is not None:
            merged.release_year_min = other.release_year_min
        if other.release_year_max is not None:
            merged.release_year_max = other.release_year_max
        if other.minimum_rating:
            merged.minimum_rating = other.minimum_rating
        if other.runtime_min is not None:
            merged.runtime_min = other.runtime_min
        if other.runtime_max is not None:
            merged.runtime_max = other.runtime_max
        if other.family_friendly:
            merged.family_friendly = True
        merged.sort_by = other.sort_by or merged.sort_by
        return merged

    def is_empty(self) -> bool:
        """True when the preference set carries no constraints at all."""
        return (
            not self.genres
            and not self.excluded_genres
            and not self.languages
            and not self.mood
            and not self.similar_to
            and not self.actors
            and not self.directors
            and not self.keywords
            and not self.country
            and self.content_type == "both"
            and self.release_year_min is None
            and self.release_year_max is None
            and not self.minimum_rating
            and self.runtime_min is None
            and self.runtime_max is None
            and not self.family_friendly
        )

    def summary(self) -> str:
        """Human-readable summary shown to the user in the sidebar."""
        parts: list[str] = []
        if self.genres:
            parts.append("Genres: " + ", ".join(self.genres))
        if self.excluded_genres:
            parts.append("Excludes: " + ", ".join(self.excluded_genres))
        if self.languages:
            parts.append("Languages: " + ", ".join(self.languages))
        if self.mood:
            parts.append(f"Mood: {self.mood}")
        if self.content_type != "both":
            parts.append(f"Type: {self.content_type}")
        if self.release_year_min or self.release_year_max:
            lo = self.release_year_min or "any"
            hi = self.release_year_max or "any"
            parts.append(f"Years: {lo} - {hi}")
        if self.minimum_rating:
            parts.append(f"Min rating: {self.minimum_rating}")
        if self.runtime_min or self.runtime_max:
            lo = self.runtime_min or "any"
            hi = self.runtime_max or "any"
            parts.append(f"Runtime (min): {lo} - {hi}")
        if self.similar_to:
            parts.append("Similar to: " + ", ".join(self.similar_to))
        if self.actors:
            parts.append("Actors: " + ", ".join(self.actors))
        if self.directors:
            parts.append("Directors: " + ", ".join(self.directors))
        if self.country:
            parts.append(f"Country: {self.country}")
        if self.family_friendly:
            parts.append("Family friendly")
        return "; ".join(parts) if parts else "No preferences set yet."


def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    """Merge two lists preserving order and dropping duplicates."""
    seen: set[str] = set()
    result: list[str] = []
    for item in [*left, *right]:
        normalized = item.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result
