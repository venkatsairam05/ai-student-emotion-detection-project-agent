"""Hybrid recommendation engine.

Combines:
    * Semantic similarity  (FAISS + embeddings over combined metadata text)
    * Metadata matching     (genres, keywords, cast, director, language, type)
    * User preference match (mood, rating, year, runtime, exclusions, people)
    * Weighted hybrid ranking (see :mod:`src.recommender.ranking`)
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from config.settings import get_settings
from src.models.movie import Movie
from src.models.preferences import UserPreferences
from src.recommender import embeddings, ranking, vector_store
from src.utils.logging_config import get_logger

logger = get_logger("recommender")

_REQUIRED_COLUMNS = (
    "id", "title", "overview", "genres", "keywords", "cast", "director",
    "rating", "popularity", "release_date", "runtime", "language",
    "content_type", "combined_text",
)

_LIST_COLUMNS = ("genres", "keywords", "cast")


class Recommender:
    """Stateless-once-loaded hybrid recommender over the local dataset."""

    def __init__(
        self,
        index=None,
        metadata: Optional[list[dict]] = None,
        movies: Optional[pd.DataFrame] = None,
    ) -> None:
        settings = get_settings()
        if index is not None:
            self.index = index
        else:
            self.index = vector_store.load_index(settings.faiss_index_path)
        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = vector_store.load_metadata(settings.faiss_metadata_path)
        if movies is not None:
            self.movies = movies
        else:
            self.movies = self._load_dataset(settings.processed_dataset_path)

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_dataset(path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
            missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                logger.warning("Processed dataset missing columns: %s", missing)
                return None
            return df
        except Exception as exc:  # pragma: no cover - corrupt data
            logger.warning("Failed to load processed dataset: %s", exc)
            return None

    @property
    def ready(self) -> bool:
        return self.movies is not None and not self.movies.empty

    @property
    def has_semantic_search(self) -> bool:
        return self.index is not None and self.metadata is not None

    # ------------------------------------------------------------------ #
    # Movie construction
    # ------------------------------------------------------------------ #
    def _to_movie(self, row: dict) -> Movie:
        return Movie(
            id=row.get("id", ""),
            title=str(row.get("title", "")),
            overview=str(row.get("overview", "") or ""),
            content_type=str(row.get("content_type", "movie")),
            genres=_parse_list(row.get("genres")),
            keywords=_parse_list(row.get("keywords")),
            cast=_parse_list(row.get("cast")),
            director=str(row.get("director", "") or ""),
            rating=_as_float(row.get("rating")),
            popularity=_as_float(row.get("popularity")),
            release_date=str(row.get("release_date", "") or ""),
            release_year=_to_int(row.get("release_year") or (row.get("release_date") or "")[:4]),
            runtime=_to_int(row.get("runtime")) or 0,
            language=str(row.get("language", "en") or "en"),
            original_language=str(row.get("original_language", "") or ""),
            poster_path=str(row.get("poster_path", "") or ""),
            tmdb_id=_to_int(row.get("tmdb_id")),
            combined_text=str(row.get("combined_text", "") or ""),
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def recommend(
        self,
        preferences: UserPreferences,
        query_text: str = "",
        top_k: Optional[int] = None,
    ) -> list[Movie]:
        """Recommend top-k titles for the given preferences + query text."""
        if not self.ready:
            return []

        source_movies = self._resolve_similar_titles(preferences.similar_to)
        excluded_ids = {str(m.id) for m in source_movies.values()}
        semantic_candidates: dict[int, float] = {}
        similarity_source: dict[int, str] = {}

        # 1) Semantic search from similar-title embeddings.
        for title, source_movie in source_movies.items():
            embed = embeddings.generate_embedding(source_movie.combined_text)
            if embed is not None and self.index is not None:
                for idx, sim in vector_store.search(
                    self.index, embed, k=30
                ):
                    if idx not in semantic_candidates or sim > semantic_candidates[idx]:
                        semantic_candidates[idx] = sim
                        similarity_source[idx] = title

        # 2) Semantic search from the free-text query.
        if query_text.strip():
            query_embed = embeddings.generate_embedding(query_text)
            if query_embed is not None and self.index is not None:
                for idx, sim in vector_store.search(
                    self.index, query_embed, k=get_settings().semantic_candidates
                ):
                    if idx not in semantic_candidates or sim > semantic_candidates[idx]:
                        semantic_candidates[idx] = sim
                        similarity_source.setdefault(idx, "")

        candidate_movies: list[Movie] = []
        for row_index, sim in semantic_candidates.items():
            movie = self._movie_from_metadata(row_index)
            if movie is None:
                continue
            if str(movie.id) in excluded_ids:
                continue
            movie.similarity_score = sim
            candidate_movies.append(movie)

        # 3) Add preference-matched rows that the semantic search missed.
        if self.movies is not None:
            candidate_movies.extend(
                self._preference_only_candidates(preferences, exclude=set(semantic_candidates))
            )

        if not candidate_movies:
            return []

        # 4) Hard metadata filtering (genres, exclusions, rating, years, runtime...).
        filtered = self.filter_candidates(candidate_movies, preferences)
        if not filtered:
            return []

        # 5) Hybrid weighted ranking.
        ranked = ranking.rank_candidates(
            filtered,
            preferences,
            top_k=top_k,
        )

        # 6) Build match reasons (explainability).
        for movie in ranked:
            src = similarity_source.get(movie.id, "")
            movie.match_reasons = ranking.build_match_reasons(
                movie, preferences, movie.similarity_score, src or None
            )
        return ranked

    def filter_candidates(
        self,
        candidates: list[Movie],
        preferences: UserPreferences,
    ) -> list[Movie]:
        """Apply hard metadata filters (return only passable movies)."""
        filtered = []
        for movie in candidates:
            if preferences.content_type in ("movie", "tv") and movie.content_type != preferences.content_type:
                continue
            if preferences.excluded_genres and set(movie.genres) & set(preferences.excluded_genres):
                continue
            if preferences.minimum_rating and movie.rating and movie.rating < preferences.minimum_rating:
                continue
            if preferences.release_year_min is not None:
                year = movie.release_year
                if year is None:
                    continue
                if year < preferences.release_year_min:
                    continue
            if preferences.release_year_max is not None:
                year = movie.release_year
                if year is None:
                    continue
                if year > preferences.release_year_max:
                    continue
            if preferences.runtime_max is not None and movie.runtime and movie.runtime > preferences.runtime_max:
                continue
            if preferences.runtime_min is not None and movie.runtime and movie.runtime < preferences.runtime_min:
                continue
            if preferences.family_friendly and set(movie.genres) & {"Horror", "Erotic", "Mature"}:
                continue
            if preferences.languages and not self._matches_language(movie, preferences.languages):
                continue
            if preferences.actors and not (set(movie.cast) & set(preferences.actors)):
                continue
            if preferences.directors and not self._matches_director(movie, preferences.directors):
                continue
            filtered.append(movie)
        return filtered

    def similar(self, title: str, top_k: Optional[int] = None) -> list[Movie]:
        """Return titles semantically similar to ``title`` in the dataset."""
        if not self.ready or self.index is None:
            return []
        source = self._find_movie_by_title(title)
        if source is None:
            return []
        embed = embeddings.generate_embedding(source.combined_text)
        if embed is None:
            return []
        k = top_k or get_settings().top_k
        results = vector_store.search(self.index, embed, k=k + 1)
        movies = []
        for idx, sim in results:
            movie = self._movie_from_metadata(idx)
            if movie is None or movie.id == source.id:
                continue
            movie.similarity_score = sim
            movies.append(movie)
        return movies[:k]

    def resolve_title(self, title: str) -> Optional[Movie]:
        """Find a dataset title by exact/fuzzy/prefix match."""
        return self._find_movie_by_title(title)

    def all_titles(self, content_type: str = "both", limit: int = 1000) -> list[str]:
        """Return dataset titles for the sidebar typeahead."""
        if not self.ready:
            return []
        df = self.movies
        if content_type in ("movie", "tv"):
            df = df[df["content_type"] == content_type]
        titles = df["title"].dropna().astype(str).str.strip()
        return sorted(set(titles))[:limit]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _resolve_similar_titles(self, titles: list[str]) -> dict[str, Movie]:
        found: dict[str, Movie] = {}
        for title in titles:
            movie = self._find_movie_by_title(title)
            if movie is not None:
                found[movie.title] = movie
        return found

    def _find_movie_by_title(self, title: str) -> Optional[Movie]:
        if self.movies is None:
            return None
        normalized = str(title).strip().lower()
        df = self.movies
        exact = df[df["title"].astype(str).str.strip().str.lower() == normalized]
        if exact.empty:
            partial = df[df["title"].astype(str).str.lower().str.contains(normalized, na=False)]
            if partial.empty:
                return None
            row = partial.iloc[0]
        else:
            row = exact.iloc[0]
        return self._to_movie(row.to_dict())

    def _movie_from_metadata(self, row_index: int) -> Optional[Movie]:
        if self.metadata is None or not (0 <= row_index < len(self.metadata)):
            return None
        return self._to_movie(self.metadata[row_index])

    def _preference_only_candidates(
        self,
        preferences: UserPreferences,
        exclude: set[int],
    ) -> list[Movie]:
        """Rows that satisfy preference filters but weren't in semantic results."""
        if self.movies is None:
            return []
        rows = [
            self._to_movie(row.to_dict())
            for _, row in self.movies.head(5000).iterrows()
        ]
        matched = self.filter_candidates(rows, preferences)
        out = []
        for movie in matched:
            row_index = self._row_index_of(movie.id)
            if row_index is None or row_index in exclude:
                continue
            out.append(movie)
        return out

    def _row_index_of(self, movie_id) -> Optional[int]:
        if self.metadata is None:
            return None
        for i, row in enumerate(self.metadata):
            if str(row.get("id")) == str(movie_id):
                return i
        return None

    @staticmethod
    def _matches_language(movie: Movie, languages: list[str]) -> bool:
        movie_lang = (movie.language or "en").lower()
        for lang in languages:
            lang = lang.lower()
            if lang == "english" and movie_lang in ("en", "en-us"):
                return True
            if movie_lang == lang:
                return True
        return False

    @staticmethod
    def _matches_director(movie: Movie, directors: list[str]) -> bool:
        director = str(movie.director or "").lower()
        return any(d.lower() in director for d in directors)


# --------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------- #
def _parse_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    else:
        raw = str(value).split("|")
    return [item.strip() for item in raw if str(item).strip()]


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
