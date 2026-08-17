"""Tests for the TMDB client (mocked HTTP layer)."""

from __future__ import annotations

from unittest import mock

import requests

from src.services.tmdb_client import TMDBClient


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_search_movie_without_key_returns_none(monkeypatch):
    from config.settings import get_settings

    monkeypatch.setenv("TMDB_API_KEY", "")
    get_settings.cache_clear()
    try:
        client = TMDBClient(api_key="")
        assert client.search_movie("Inception") is None
    finally:
        get_settings.cache_clear()


def test_search_movie_success(mock_tmdb_get):
    client = TMDBClient(api_key="fake-key")
    mock_tmdb_get.return_value = _FakeResponse(
        {"results": [{"id": 1, "title": "Inception"}]}
    )
    result = client.search_movie("Inception")
    assert result["results"][0]["id"] == 1
    mock_tmdb_get.assert_called_once()


def test_error_handling_returns_none(mock_tmdb_get):
    client = TMDBClient(api_key="fake-key", retries=1)
    mock_tmdb_get.side_effect = requests.ConnectionError("network down")
    with mock.patch("time.sleep", return_value=None):
        assert client.get_movie_details(1) is None


def test_http_error_returns_none(mock_tmdb_get):
    client = TMDBClient(api_key="fake-key", retries=0)
    mock_tmdb_get.return_value = _FakeResponse({"error": "not found"}, status_code=404)
    assert client.get_movie_details(999) is None


def test_rate_limit_retries(mock_tmdb_get):
    client = TMDBClient(api_key="fake-key", retries=2, rate_limit_sleep=0.0)
    mock_tmdb_get.side_effect = [
        _FakeResponse({}, status_code=429),
        _FakeResponse({"results": [{"id": 5}]}),
    ]
    with mock.patch("time.sleep", return_value=None):
        result = client.search_tv("Stranger Things")
    assert result["results"][0]["id"] == 5
    assert mock_tmdb_get.call_count == 2


def test_caching_prevents_duplicate_calls(mock_tmdb_get):
    client = TMDBClient(api_key="fake-key")
    mock_tmdb_get.return_value = _FakeResponse({"results": [{"id": 7}]})
    client.search_movie("Memento")
    client.search_movie("Memento")
    assert mock_tmdb_get.call_count == 1


def test_enrich_resolves_and_fills_fields(mock_tmdb_get):
    from src.models.movie import Movie

    client = TMDBClient(api_key="fake-key")
    movie = Movie(id="x", title="Inception", content_type="movie", tmdb_id=27205)
    mock_tmdb_get.side_effect = [
        _FakeResponse({"poster_path": "/abc.jpg", "runtime": 148, "vote_average": 8.8}),
        _FakeResponse({"results": [{"site": "YouTube", "type": "Trailer", "key": "YoHD9XEInc0"}]}),
    ]
    enriched = client.enrich(movie)
    assert enriched.poster_path == "/abc.jpg"
    assert enriched.runtime == 148
    assert enriched.trailer_key == "YoHD9XEInc0"


def test_enrich_writes_back_resolved_tmdb_id(mock_tmdb_get):
    from src.models.movie import Movie

    client = TMDBClient(api_key="fake-key")
    movie = Movie(id="x", title="Inception", content_type="movie")
    mock_tmdb_get.side_effect = [
        _FakeResponse({"results": [{"id": 27205}]}),
        _FakeResponse({"poster_path": "/abc.jpg", "runtime": 148, "vote_average": 8.8}),
        _FakeResponse({"results": [{"site": "YouTube", "type": "Trailer", "key": "YoHD9XEInc0"}]}),
    ]
    enriched = client.enrich(movie)
    assert enriched.tmdb_id == 27205
    assert enriched.poster_path == "/abc.jpg"
    assert enriched.trailer_key == "YoHD9XEInc0"


def test_poster_url():
    client = TMDBClient(api_key="")
    assert client.poster_url("") == ""
    url = client.poster_url("/abc.jpg")
    assert url.endswith("/abc.jpg")


def test_backdrop_url_uses_w1280_size():
    client = TMDBClient(api_key="")
    assert client.backdrop_url("") == ""
    url = client.backdrop_url("/back.jpg")
    assert "/w1280/back.jpg" in url


def test_trending(mock_tmdb_get):
    client = TMDBClient(api_key="fake-key")
    mock_tmdb_get.return_value = _FakeResponse({"results": [{"id": 11}]})
    assert client.trending()["results"][0]["id"] == 11
    url = mock_tmdb_get.call_args.kwargs["params"]["api_key"]
    assert url == "fake-key"


def test_trending_passes_media_and_window(mock_tmdb_get):
    client = TMDBClient(api_key="fake-key")
    mock_tmdb_get.return_value = _FakeResponse({"results": []})
    client.trending("movie", "day")
    request_url = mock_tmdb_get.call_args[0][0]
    assert request_url.endswith("/trending/movie/day")


def test_top_rated_and_popular_rows(mock_tmdb_get):
    client = TMDBClient(api_key="fake-key")
    mock_tmdb_get.return_value = _FakeResponse({"results": [{"id": 21}]})
    assert client.top_rated_movies()["results"][0]["id"] == 21
    assert client.popular_movies()["results"][0]["id"] == 21
    assert client.popular_tv()["results"][0]["id"] == 21
    assert client.top_rated_tv()["results"][0]["id"] == 21
    urls = [c.args[0] for c in mock_tmdb_get.call_args_list]
    assert any(u.endswith("/movie/top_rated") for u in urls)
    assert any(u.endswith("/movie/popular") for u in urls)
    assert any(u.endswith("/tv/popular") for u in urls)
    assert any(u.endswith("/tv/top_rated") for u in urls)


# --------------------------------------------------------------------- #
# Spec-facing method names + Bearer auth
# --------------------------------------------------------------------- #
def test_bearer_token_uses_authorization_header(mock_tmdb_get):
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefghijklmnopqrstuvwxyz"
    client = TMDBClient(api_key=jwt)
    mock_tmdb_get.return_value = _FakeResponse({"results": [{"id": 27205}]})
    client.search_movie("Inception")
    request = mock_tmdb_get.call_args
    assert request.kwargs["headers"].get("Authorization") == f"Bearer {jwt}"
    assert "api_key" not in request.kwargs.get("params", {})


def test_v3_key_still_uses_api_key_param(mock_tmdb_get):
    client = TMDBClient(api_key="0123456789abcdef0123456789abcdef")
    mock_tmdb_get.return_value = _FakeResponse({"results": [{"id": 1}]})
    client.search_movie("Inception")
    request = mock_tmdb_get.call_args
    assert request.kwargs["params"]["api_key"] == "0123456789abcdef0123456789abcdef"
    assert "Authorization" not in request.kwargs.get("headers", {})


def test_spec_methods_hit_correct_endpoints(mock_tmdb_get):
    client = TMDBClient(api_key="fake-key")
    mock_tmdb_get.return_value = _FakeResponse({"results": []})

    client.search_movies("Memento")
    client.get_similar_movies(77)
    client.get_similar_tv(1399)
    client.get_movie_recommendations(27205)
    client.get_tv_recommendations(1396)
    client.get_trending_movies()
    client.get_trending_tv()
    client.get_popular_movies()
    client.get_popular_tv()
    client.get_top_rated_movies()
    client.get_top_rated_tv()

    urls = [c.args[0] for c in mock_tmdb_get.call_args_list]
    assert any(u.endswith("/search/movie") for u in urls)
    assert any(u.endswith("/movie/77/similar") for u in urls)
    assert any(u.endswith("/tv/1399/similar") for u in urls)
    assert any(u.endswith("/movie/27205/recommendations") for u in urls)
    assert any(u.endswith("/tv/1396/recommendations") for u in urls)
    assert any(u.endswith("/trending/movie/week") for u in urls)
    assert any(u.endswith("/trending/tv/week") for u in urls)
    assert any(u.endswith("/movie/popular") for u in urls)
    assert any(u.endswith("/tv/popular") for u in urls)
    assert any(u.endswith("/movie/top_rated") for u in urls)
    assert any(u.endswith("/tv/top_rated") for u in urls)


def test_search_tv_success(mock_tmdb_get):
    client = TMDBClient(api_key="fake-key")
    mock_tmdb_get.return_value = _FakeResponse({"results": [{"id": 31}]})
    result = client.search_tv("Severance")
    assert result["results"][0]["id"] == 31


def test_details_credits_videos_similar(mock_tmdb_get):
    client = TMDBClient(api_key="fake-key")
    mock_tmdb_get.return_value = _FakeResponse({"id": 41})
    assert client.get_tv_details(41)["id"] == 41
    assert client.get_movie_credits(41)["id"] == 41
    assert client.get_tv_credits(41)["id"] == 41
    assert client.get_movie_videos(41)["id"] == 41
    assert client.get_tv_videos(41)["id"] == 41
    assert client.similar_movies(41)["id"] == 41
    assert client.similar_tv(41)["id"] == 41
    urls = [c.args[0] for c in mock_tmdb_get.call_args_list]
    assert any(u.endswith("/tv/41") for u in urls)
    assert any(u.endswith("/movie/41/credits") for u in urls)
    assert any(u.endswith("/movie/41/similar") for u in urls)


def test_genres_and_watch_providers(mock_tmdb_get):
    client = TMDBClient(api_key="fake-key")
    mock_tmdb_get.return_value = _FakeResponse({"genres": []})
    assert client.movie_genres()["genres"] == []
    assert client.tv_genres()["genres"] == []
    assert client.watch_providers("movie", 51)["genres"] == []
    urls = [c.args[0] for c in mock_tmdb_get.call_args_list]
    assert any(u.endswith("/genre/movie/list") for u in urls)
    assert any(u.endswith("/movie/51/watch/providers") for u in urls)


def test_pick_trailer_prefers_youtube_official_trailer():
    results = [
        {"site": "YouTube", "type": "Teaser", "key": "aa", "official": True},
        {"site": "YouTube", "type": "Trailer", "key": "bb", "official": True},
        {"site": "Vimeo", "type": "Trailer", "key": "cc", "official": True},
    ]
    assert TMDBClient._pick_trailer(results) == "bb"
    assert TMDBClient._pick_trailer([{"site": "Vimeo", "type": "Trailer", "key": "cc"}]) == ""
    assert TMDBClient._pick_trailer([]) == ""


def test_search_multi(mock_tmdb_get):
    client = TMDBClient(api_key="fake-key")
    mock_tmdb_get.return_value = _FakeResponse({"results": [{"id": 1, "media_type": "movie"}]})
    result = client.search_multi("Inception")
    assert result["results"][0]["media_type"] == "movie"
    assert mock_tmdb_get.call_args[0][0].endswith("/search/multi")


def test_search_person_and_combined_credits(mock_tmdb_get):
    client = TMDBClient(api_key="fake-key")
    mock_tmdb_get.side_effect = [
        _FakeResponse({"results": [{"id": 31, "name": "Tom Hanks", "popularity": 50.0}]}),
        _FakeResponse({"cast": [{"id": 5, "media_type": "movie", "title": "Forrest Gump"}]}),
    ]
    result = client.search_person("Tom Hanks")
    assert result["results"][0]["id"] == 31
    credits = client.person_combined_credits(31)
    assert credits["cast"][0]["title"] == "Forrest Gump"
    assert mock_tmdb_get.call_args[0][0].endswith("/person/31/combined_credits")


def test_genre_names_resolves_and_caches(mock_tmdb_get):
    client = TMDBClient(api_key="fake-key")
    mock_tmdb_get.return_value = _FakeResponse(
        {"genres": [{"id": 28, "name": "Action"}, {"id": 878, "name": "Science Fiction"}]}
    )
    assert client.genre_names([28, 878, 999], "movie") == ["Action", "Science Fiction"]
    assert client.genre_names([28], "movie") == ["Action"]
    # Genre list fetched only once (cached).
    assert mock_tmdb_get.call_count == 1


def test_tmdb_url():
    client = TMDBClient(api_key="")
    assert client.tmdb_url("movie", 27205) == "https://www.themoviedb.org/movie/27205"
    assert client.tmdb_url("tv", 1396) == "https://www.themoviedb.org/tv/1396"
    assert client.tmdb_url("movie", None) is None
    assert client.tmdb_url("tv", 0) is None


def test_image_url_helpers():
    from src.services.tmdb_client import get_backdrop_url, get_poster_url

    assert get_poster_url(None) is None
    assert get_poster_url("") is None
    poster = get_poster_url("/abc.jpg")
    assert poster is not None and poster.startswith("https://image.tmdb.org/t/p/")
    assert poster.endswith("/w500/abc.jpg")
    hi = get_poster_url("/abc.jpg", "w780")
    assert hi.endswith("/w780/abc.jpg")
    backdrop = get_backdrop_url("/bg.jpg")
    assert backdrop.endswith("/original/bg.jpg")
    assert get_backdrop_url(None) is None


def test_search_normalizes_movies_and_tv(mock_tmdb_get):
    client = TMDBClient(api_key="fake-key")

    def _fake(url, **kwargs):
        if url.endswith("/search/multi"):
            return _FakeResponse({"results": [
                {"id": 1, "media_type": "movie", "title": "Inception", "overview": "Dream heist",
                 "poster_path": "/a.jpg", "backdrop_path": "/b.jpg", "genre_ids": [28, 878],
                 "vote_average": 8.8, "release_date": "2010-07-16", "popularity": 80.0},
                {"id": 2, "media_type": "tv", "name": "Breaking Bad", "overview": "Chem teacher",
                 "genre_ids": [18], "vote_average": 9.5, "first_air_date": "2008-01-20", "popularity": 90.0},
            ]})
        if url.endswith("/genre/movie/list"):
            return _FakeResponse({"genres": [{"id": 28, "name": "Action"}, {"id": 878, "name": "Science Fiction"}]})
        if url.endswith("/genre/tv/list"):
            return _FakeResponse({"genres": [{"id": 18, "name": "Drama"}]})
        return _FakeResponse({"results": []})

    mock_tmdb_get.side_effect = _fake
    movies, ok = client.search("Inception", enrich_limit=0)
    assert ok is True
    assert [m.title for m in movies] == ["Breaking Bad", "Inception"]  # sorted by popularity
    movie = next(m for m in movies if m.content_type == "movie")
    assert movie.genres == ["Action", "Science Fiction"]
    assert movie.release_year == 2010
    assert movie.backdrop_path == "/b.jpg"
    tv = next(m for m in movies if m.content_type == "tv")
    assert tv.genres == ["Drama"]
    assert tv.release_year == 2008


def test_search_default_enrich_limit_does_not_raise(mock_tmdb_get):
    client = TMDBClient(api_key="fake-key")

    def _fake(url, **kwargs):
        if url.endswith("/search/multi"):
            return _FakeResponse({"results": [
                {"id": 27205, "media_type": "movie", "title": "Inception",
                 "overview": "Dream heist", "genre_ids": [878],
                 "vote_average": 8.8, "release_date": "2010-07-16", "popularity": 80.0},
            ]})
        if url.endswith("/genre/movie/list"):
            return _FakeResponse({"genres": [{"id": 878, "name": "Science Fiction"}]})
        return _FakeResponse({"results": []})

    mock_tmdb_get.side_effect = _fake
    movies, ok = client.search("Inception")  # default enrich_limit, live path
    assert ok is True
    assert [m.title for m in movies] == ["Inception"]


def test_search_includes_person_credits(mock_tmdb_get):
    client = TMDBClient(api_key="fake-key")

    def _fake(url, **kwargs):
        if url.endswith("/search/multi"):
            return _FakeResponse({"results": [
                {"id": 9, "media_type": "person", "name": "Tom Hanks", "popularity": 50.0},
            ]})
        if url.endswith("/search/person"):
            return _FakeResponse({"results": [{"id": 9, "name": "Tom Hanks", "popularity": 50.0}]})
        if url.endswith("/person/9/combined_credits"):
            return _FakeResponse({"cast": [
                {"id": 101, "media_type": "movie", "title": "Forrest Gump", "genre_ids": [18],
                 "vote_average": 8.8, "release_date": "1994-07-06", "popularity": 70.0},
                {"id": 102, "media_type": "tv", "name": "Band of Brothers", "genre_ids": [18],
                 "vote_average": 9.4, "first_air_date": "2001-09-09", "popularity": 60.0},
            ]})
        if url.endswith("/genre/movie/list"):
            return _FakeResponse({"genres": [{"id": 18, "name": "Drama"}]})
        if url.endswith("/genre/tv/list"):
            return _FakeResponse({"genres": [{"id": 18, "name": "Drama"}]})
        return _FakeResponse({"results": []})

    mock_tmdb_get.side_effect = _fake
    movies, ok = client.search("Tom Hanks", enrich_limit=0)
    assert ok is True
    titles = {m.title for m in movies}
    assert "Forrest Gump" in titles
    assert "Band of Brothers" in titles


def test_search_failure_returns_ok_false(mock_tmdb_get):
    client = TMDBClient(api_key="fake-key", retries=0)
    mock_tmdb_get.side_effect = requests.ConnectionError("network down")
    movies, ok = client.search("Inception", enrich_limit=0)
    assert movies == []
    assert ok is False
