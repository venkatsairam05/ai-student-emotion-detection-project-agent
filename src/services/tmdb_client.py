"""TMDB API client with caching, retries and rate-limit handling.

The client is fully optional: every public method returns ``None`` when the
API key is missing or the upstream call fails, so the application degrades to
local dataset + placeholder behaviour instead of crashing.
"""

from __future__ import annotations

import re
import time
from typing import Any, Optional

import requests

from config.settings import get_settings
from src.models.movie import Movie
from src.utils.cache import TTLCache
from src.utils.logging_config import get_logger

logger = get_logger("tmdb")

# A TMDB v4 "read access token" is a JWT (three dot-separated base64url
# segments starting with "eyJ"). The API authenticates such tokens via the
# ``Authorization: Bearer <token>`` header, while v3 API keys go in the query
# string as ``api_key``.
_JWT_TOKEN_RE = re.compile(r"^eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*$")


# --------------------------------------------------------------------- #
# Image CDN helpers (TMDB poster / backdrop URLs)
# --------------------------------------------------------------------- #
def _image_cdn_base() -> str:
    """Root TMDB image CDN (``https://image.tmdb.org/t/p``)."""
    base = get_settings().tmdb_image_base_url.rstrip("/")
    if re.search(r"/w\d+$", base):
        base = base.rsplit("/", 1)[0]
    return base


def get_poster_url(poster_path: Optional[str], size: str = "w500") -> Optional[str]:
    """Build a TMDB poster URL (``w500`` by default, ``w780`` for hi-res)."""
    if not poster_path:
        return None
    return f"{_image_cdn_base()}/{size}{poster_path}"


def get_backdrop_url(backdrop_path: Optional[str], size: str = "original") -> Optional[str]:
    """Build a TMDB backdrop URL (``original`` by default, ``w1280`` for smaller)."""
    if not backdrop_path:
        return None
    return f"{_image_cdn_base()}/{size}{backdrop_path}"


def _first_year(item: dict) -> Optional[int]:
    """Pull the release/first-air year out of a TMDB item (int, or None)."""
    for key in ("release_date", "first_air_date"):
        value = item.get(key) or ""
        if len(value) >= 4 and value[:4].isdigit():
            return int(value[:4])
    return None


class TMDBError(Exception):
    """Raised when the TMDB API returns an unexpected result."""


class TMDBClient:
    """Thin, defensive wrapper around the TMDB REST API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        retries: Optional[int] = None,
        rate_limit_sleep: Optional[float] = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.tmdb_api_key
        self.base_url = settings.tmdb_base_url
        self.timeout = settings.tmdb_timeout_seconds
        self.retries = retries if retries is not None else settings.tmdb_retries
        self.rate_limit_sleep = (
            rate_limit_sleep if rate_limit_sleep is not None else settings.tmdb_rate_limit_sleep
        )
        # Read-access tokens (JWT) authenticate via the Authorization header;
        # classic v3 keys authenticate via the ``api_key`` query parameter.
        self._uses_bearer_auth: bool = bool(self.api_key) and _JWT_TOKEN_RE.match(
            self.api_key
        ) is not None
        self._cache: TTLCache = TTLCache(max_size=512, ttl_seconds=6 * 3600)

    # ------------------------------------------------------------------ #
    # Low level request helpers
    # ------------------------------------------------------------------ #
    def _get(self, endpoint: str, params: dict[str, Any]) -> Optional[dict]:
        """GET ``endpoint`` with retry + basic rate-limit handling."""
        if not self.api_key:
            logger.debug("TMDB API key missing; skipping request to %s", endpoint)
            return None
        cache_key = endpoint + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        query = dict(params)
        headers: dict[str, str] = {}
        if self._uses_bearer_auth:
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            query["api_key"] = self.api_key

        for attempt in range(self.retries + 1):
            try:
                response = requests.get(
                    url, params=query, headers=headers, timeout=self.timeout
                )
                if response.status_code == 429:  # rate limited
                    time.sleep(self.rate_limit_sleep * (attempt + 1))
                    continue
                response.raise_for_status()
                payload = response.json()
                self._cache.set(cache_key, payload)
                return payload
            except requests.RequestException as exc:
                logger.warning(
                    "TMDB request failed (attempt %s) for %s: %s",
                    attempt + 1,
                    endpoint,
                    exc,
                )
                if attempt >= self.retries:
                    break
                time.sleep(0.5 * (attempt + 1))
        return None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def search_movie(self, query: str, year: int | None = None) -> Optional[dict]:
        params: dict[str, Any] = {"query": query, "include_adult": "false"}
        if year:
            params["year"] = year
        return self._get("/search/movie", params)

    def search_movies(self, query: str, year: int | None = None) -> Optional[dict]:
        """Alias of :meth:`search_movie` (spec-facing name)."""
        return self.search_movie(query, year)

    def search_tv(self, query: str, first_air_date_year: int | None = None) -> Optional[dict]:
        params: dict[str, Any] = {"query": query, "include_adult": "false"}
        if first_air_date_year:
            params["first_air_date_year"] = first_air_date_year
        return self._get("/search/tv", params)

    def search_multi(self, query: str) -> Optional[dict]:
        """Search movies, TV shows and people in one request."""
        return self._get("/search/multi", {"query": query, "include_adult": "false"})

    def search_person(self, query: str) -> Optional[dict]:
        return self._get("/search/person", {"query": query, "include_adult": "false"})

    def person_combined_credits(self, person_id: int) -> Optional[dict]:
        """Movies/TV shows a person has appeared in / worked on."""
        return self._get(f"/person/{person_id}/combined_credits", {"language": "en-US"})

    def get_movie_details(self, movie_id: int) -> Optional[dict]:
        return self._get(f"/movie/{movie_id}", {"language": "en-US"})

    def get_tv_details(self, tv_id: int) -> Optional[dict]:
        return self._get(f"/tv/{tv_id}", {"language": "en-US"})

    def get_movie_credits(self, movie_id: int) -> Optional[dict]:
        return self._get(f"/movie/{movie_id}/credits", {"language": "en-US"})

    def get_tv_credits(self, tv_id: int) -> Optional[dict]:
        return self._get(f"/tv/{tv_id}/credits", {"language": "en-US"})

    def get_movie_videos(self, movie_id: int) -> Optional[dict]:
        return self._get(f"/movie/{movie_id}/videos", {"language": "en-US"})

    def get_tv_videos(self, tv_id: int) -> Optional[dict]:
        return self._get(f"/tv/{tv_id}/videos", {"language": "en-US"})

    def similar_movies(self, movie_id: int) -> Optional[dict]:
        return self._get(f"/movie/{movie_id}/similar", {"language": "en-US"})

    def similar_tv(self, tv_id: int) -> Optional[dict]:
        return self._get(f"/tv/{tv_id}/similar", {"language": "en-US"})

    def get_similar_movies(self, movie_id: int) -> Optional[dict]:
        """Alias of :meth:`similar_movies` (spec-facing name)."""
        return self.similar_movies(movie_id)

    def get_similar_tv(self, tv_id: int) -> Optional[dict]:
        """Alias of :meth:`similar_tv` (spec-facing name)."""
        return self.similar_tv(tv_id)

    def get_movie_recommendations(self, movie_id: int) -> Optional[dict]:
        return self._get(f"/movie/{movie_id}/recommendations", {"language": "en-US"})

    def get_tv_recommendations(self, tv_id: int) -> Optional[dict]:
        return self._get(f"/tv/{tv_id}/recommendations", {"language": "en-US"})

    def get_trending_movies(self, time_window: str = "week") -> Optional[dict]:
        """Alias of :meth:`trending` for movies (spec-facing name)."""
        return self.trending("movie", time_window)

    def get_trending_tv(self, time_window: str = "week") -> Optional[dict]:
        """Alias of :meth:`trending` for TV shows (spec-facing name)."""
        return self.trending("tv", time_window)

    def get_popular_movies(self) -> Optional[dict]:
        """Alias of :meth:`popular_movies` (spec-facing name)."""
        return self.popular_movies()

    def get_popular_tv(self) -> Optional[dict]:
        """Alias of :meth:`popular_tv` (spec-facing name)."""
        return self.popular_tv()

    def get_top_rated_movies(self) -> Optional[dict]:
        """Alias of :meth:`top_rated_movies` (spec-facing name)."""
        return self.top_rated_movies()

    def get_top_rated_tv(self) -> Optional[dict]:
        """Alias of :meth:`top_rated_tv` (spec-facing name)."""
        return self.top_rated_tv()

    def trending(self, media_type: str = "all", time_window: str = "week") -> Optional[dict]:
        return self._get(f"/trending/{media_type}/{time_window}", {"language": "en-US"})

    def popular_movies(self) -> Optional[dict]:
        return self._get("/movie/popular", {"language": "en-US"})

    def top_rated_movies(self) -> Optional[dict]:
        return self._get("/movie/top_rated", {"language": "en-US"})

    def popular_tv(self) -> Optional[dict]:
        return self._get("/tv/popular", {"language": "en-US"})

    def top_rated_tv(self) -> Optional[dict]:
        return self._get("/tv/top_rated", {"language": "en-US"})

    def movie_genres(self) -> Optional[dict]:
        return self._get("/genre/movie/list", {"language": "en-US"})

    def tv_genres(self) -> Optional[dict]:
        return self._get("/genre/tv/list", {"language": "en-US"})

    def watch_providers(self, media_type: str, item_id: int) -> Optional[dict]:
        return self._get(f"/{media_type}/{item_id}/watch/providers", {"language": "en-US"})

    # ------------------------------------------------------------------ #
    # Genre resolution (movie / tv genre id -> name, cached)
    # ------------------------------------------------------------------ #
    def genre_names(self, genre_ids: Optional[list], media_type: str = "movie") -> list[str]:
        """Map TMDB ``genre_ids`` to human-readable names (cached lists)."""
        mapping = self._load_genres(media_type)
        names: list[str] = []
        for gid in genre_ids or []:
            try:
                name = mapping.get(int(gid))
            except (TypeError, ValueError):
                name = None
            if name:
                names.append(name)
        return names

    def _load_genres(self, media_type: str = "movie") -> dict[int, str]:
        cache_key = f"__genres_{media_type}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        payload = self.movie_genres() if media_type == "movie" else self.tv_genres()
        mapping: dict[int, str] = {}
        for genre in (payload or {}).get("genres") or []:
            try:
                mapping[int(genre["id"])] = genre["name"]
            except (KeyError, TypeError, ValueError):
                continue
        self._cache.set(cache_key, mapping)
        return mapping

    # ------------------------------------------------------------------ #
    # High-level search (movies + TV + people), normalized to Movie models
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        limit: int = 20,
        enrich_limit: Optional[int] = None,
    ) -> tuple[list[Movie], bool]:
        """Search movies/TV shows (and, for actor/director queries, a person's
        credits) and return normalized :class:`Movie` models.

        Returns ``(movies, ok)`` where ``ok`` is ``False`` only when the
        primary TMDB call failed (network/API error) so the UI can distinguish
        "no results" from "service error".
        """
        movies, ok = self._search_normalized(query, limit)
        if not ok:
            return movies, False
        cap = get_settings().tmdb_search_enrich_limit if enrich_limit is None else enrich_limit
        for movie in movies[:cap]:
            self.enrich(movie)
        return movies, True

    def _search_normalized(self, query: str, limit: int) -> tuple[list[Movie], bool]:
        multi = self.search_multi(query)
        if multi is None:
            return [], False
        results = multi.get("results") or []

        movies: list[Movie] = []
        seen: set[tuple[str, Any]] = set()
        for item in results:
            media_type = item.get("media_type")
            if media_type in ("movie", "tv"):
                movie = self._movie_from_tmdb(item, media_type)
                key = (media_type, movie.tmdb_id)
                if key not in seen:
                    seen.add(key)
                    movies.append(movie)

        persons = [p for p in results if p.get("media_type") == "person"]
        if not persons:
            person_payload = self.search_person(query)
            persons = (person_payload or {}).get("results") or []
        persons = sorted(persons, key=lambda p: p.get("popularity") or 0, reverse=True)
        for person in persons[:1]:
            credits = self.person_combined_credits(person.get("id")) or {}
            entries = sorted(
                (credits.get("cast") or []) + (credits.get("crew") or []),
                key=lambda c: c.get("popularity") or 0,
                reverse=True,
            )
            for item in entries:
                media_type = item.get("media_type")
                if media_type in ("movie", "tv"):
                    movie = self._movie_from_tmdb(item, media_type)
                    key = (media_type, movie.tmdb_id)
                    if key not in seen:
                        seen.add(key)
                        movies.append(movie)

        movies.sort(key=lambda m: m.popularity, reverse=True)
        return movies[:limit], True

    def _movie_from_tmdb(self, item: dict, media_type: str) -> Movie:
        """Normalize a TMDB search/credits item into the existing Movie model."""
        return Movie(
            id=item.get("id") or 0,
            title=item.get("title") or item.get("name") or "Untitled",
            overview=item.get("overview") or "",
            content_type=media_type,
            genres=self.genre_names(item.get("genre_ids"), media_type),
            rating=float(item.get("vote_average") or 0.0),
            popularity=float(item.get("popularity") or 0.0),
            release_date=item.get("release_date") or item.get("first_air_date") or "",
            release_year=_first_year(item),
            poster_path=item.get("poster_path") or "",
            backdrop_path=item.get("backdrop_path") or "",
            tmdb_id=item.get("id"),
        )

    def tmdb_url(self, media_type: str, tmdb_id: Any) -> Optional[str]:
        """Official TMDB page URL for a movie or TV show (no fabricated URLs)."""
        if not tmdb_id:
            return None
        section = "tv" if media_type == "tv" else "movie"
        return f"https://www.themoviedb.org/{section}/{tmdb_id}"

    # ------------------------------------------------------------------ #
    # Convenience: enrich a Movie with live TMDB data (poster, trailer)
    # ------------------------------------------------------------------ #
    def enrich(self, movie: Any) -> Any:
        """Best-effort enrichment of a Movie with poster/trailer URLs.

        Never raises and never blocks the app for long.
        """
        try:
            tmdb_id = self._resolve_tmdb_id(movie)
            if tmdb_id is None:
                return movie
            movie.tmdb_id = tmdb_id
            if movie.content_type == "tv":
                details = self.get_tv_details(tmdb_id)
                videos = self.get_tv_videos(tmdb_id)
            else:
                details = self.get_movie_details(tmdb_id)
                videos = self.get_movie_videos(tmdb_id)
            if details:
                movie.poster_path = details.get("poster_path") or movie.poster_path
                movie.backdrop_path = details.get("backdrop_path") or movie.backdrop_path
                if details.get("runtime"):
                    movie.runtime = details.get("runtime")
                movie.rating = float(details.get("vote_average") or movie.rating or 0.0)
                movie.language = details.get("original_language") or movie.language
                genre_names = [g.get("name") for g in (details.get("genres") or []) if g.get("name")]
                if genre_names:
                    movie.genres = genre_names
            if videos and videos.get("results"):
                movie.trailer_key = self._pick_trailer(videos["results"])
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("TMDB enrichment failed for %s: %s", movie.title, exc)
        return movie

    def poster_url(self, poster_path: str) -> str:
        return get_poster_url(poster_path, "w500") or ""

    def backdrop_url(self, backdrop_path: str, size: str = "w1280") -> str:
        return get_backdrop_url(backdrop_path, size) or ""

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _resolve_tmdb_id(self, movie: Any) -> Optional[int]:
        if movie.tmdb_id:
            return int(movie.tmdb_id)
        try:
            if movie.content_type == "tv":
                payload = self.search_tv(movie.title)
                key = "results"
            else:
                payload = self.search_movie(movie.title)
                key = "results"
            if not payload or not payload.get(key):
                return None
            return int(payload[key][0]["id"])
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            logger.debug("Could not resolve TMDB id for %s: %s", movie.title, exc)
            return None

    @staticmethod
    def _pick_trailer(results: list[dict]) -> str:
        for item in sorted(
            results,
            key=lambda r: (
                r.get("site") == "YouTube",
                r.get("type") == "Trailer",
                r.get("official", False),
            ),
            reverse=True,
        ):
            if item.get("site") == "YouTube" and item.get("key"):
                return item["key"]
        return ""
