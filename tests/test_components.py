"""Tests for pure card-normalization helpers (no widgets involved)."""

from __future__ import annotations

from src.ui import components


class _FakeMovie:
    def __init__(self):
        self.title = "Inception"
        self.overview = "A thief steals secrets from dreams."
        self.rating = 8.8
        self.display_year = "2010"
        self.genres = ["Action", "Science Fiction"]
        self.poster_path = "/inception.jpg"
        self.backdrop_path = "/inception_bg.jpg"
        self.runtime = 148
        self.content_type = "movie"
        self.director = "Christopher Nolan"
        self.cast = ["Leonardo DiCaprio"]
        self.trailer_key = "YoHD9XEInc0"
        self.tmdb_id = 27205
        self.final_score = 0.93
        self.match_reasons = ["You love sci-fi thrillers"]


class _FakeTMDB:
    def poster_url(self, path):
        return f"https://img.example{path}"

    def backdrop_url(self, path):
        return f"https://img.example/w1280{path}"

    def genre_names(self, genre_ids, media):
        mapping = {28: "Action", 878: "Science Fiction", 18: "Drama", 10765: "Sci-Fi & Fantasy"}
        return [mapping[int(g)] for g in (genre_ids or []) if int(g) in mapping]


def test_movie_to_card_normalizes_fields():
    card = components.movie_to_card(_FakeMovie(), _FakeTMDB())
    assert card["title"] == "Inception"
    assert card["media"] == "movie"
    assert card["content_type"] == "movie"
    assert card["tmdb_id"] == 27205
    assert card["match_pct"] == 93
    assert card["poster"] == "https://img.example/inception.jpg"
    assert card["backdrop"] == "https://img.example/w1280/inception_bg.jpg"
    assert card["trailer_key"] == "YoHD9XEInc0"
    assert card["tmdb_url"] == "https://www.themoviedb.org/movie/27205"


def test_movie_to_card_no_tmdb_keeps_urls_blank():
    card = components.movie_to_card(_FakeMovie(), None)
    assert card["poster"] == ""
    assert card["backdrop"] == ""


def test_tv_show_card_media_is_tv():
    movie = _FakeMovie()
    movie.content_type = "tv"
    movie.display_year = "2021"
    movie.tmdb_id = 1396
    card = components.movie_to_card(movie, None)
    assert card["media"] == "tv"
    assert card["content_type"] == "tv"
    assert card["tmdb_url"] == "https://www.themoviedb.org/tv/1396"


def test_movie_to_card_no_tmdb_id_has_no_url():
    movie = _FakeMovie()
    movie.tmdb_id = None
    card = components.movie_to_card(movie, _FakeTMDB())
    assert card["tmdb_url"] == ""


def test_tmdb_item_to_card_movie():
    item = {
        "id": 123,
        "title": "Parasite",
        "overview": "Class satire",
        "vote_average": 8.5,
        "release_date": "2019-05-30",
        "poster_path": "/p.jpg",
    }
    card = components.tmdb_item_to_card(item, _FakeTMDB(), "movie")
    assert card["title"] == "Parasite"
    assert card["media"] == "movie"
    assert card["year"] == "2019"
    assert card["rating"] == 8.5
    assert card["tmdb_id"] == 123
    assert card["match_pct"] is None


def test_tmdb_item_to_card_tv_uses_name_and_first_air_date():
    item = {
        "id": 456,
        "name": "Severance",
        "overview": "",
        "vote_average": 8.7,
        "first_air_date": "2022-02-18",
        "poster_path": None,
    }
    card = components.tmdb_item_to_card(item, None, "tv")
    assert card["title"] == "Severance"
    assert card["media"] == "tv"
    assert card["year"] == "2022"
    assert card["poster"] == ""


def test_tmdb_item_to_card_missing_fields_are_tolerated():
    card = components.tmdb_item_to_card({}, None, "movie")
    assert card["title"] == "Untitled"
    assert card["year"] == "N/A"
    assert card["genres"] == []


def test_tmdb_item_to_card_resolves_genres_via_client():
    item = {
        "id": 27205,
        "title": "Inception",
        "genre_ids": [28, 878],
        "poster_path": "/in.jpg",
    }
    card = components.tmdb_item_to_card(item, _FakeTMDB(), "movie")
    assert card["genres"] == ["Action", "Science Fiction"]
    assert card["tmdb_url"] == "https://www.themoviedb.org/movie/27205"


def test_tmdb_item_to_card_tv_genres():
    item = {
        "id": 60735,
        "name": "The Flash",
        "genre_ids": [10765, 18],
        "first_air_date": "2014-10-07",
        "poster_path": None,
    }
    card = components.tmdb_item_to_card(item, _FakeTMDB(), "tv")
    assert card["genres"] == ["Sci-Fi & Fantasy", "Drama"]
    assert card["year"] == "2014"
    assert card["tmdb_url"] == "https://www.themoviedb.org/tv/60735"


def test_movie_card_html_escapes_user_data_and_omits_empty_pieces():
    card = {
        "title": "Tom & Jerry <3",
        "overview": "A cat & mouse <b>rivalry</b>.",
        "rating": 8.5,
        "year": "2021",
        "genres": ["Comedy"],
        "poster": "",
        "runtime": None,
        "content_type": "movie",
        "trailer_key": "",
        "tmdb_url": "",
        "match_pct": None,
        "reasons": [],
    }
    html = components.movie_card_html(card)
    assert "Tom &amp; Jerry &lt;3" in html
    assert "cat &amp; mouse" in html
    assert "<b>rivalry</b>" not in html
    assert "▶ Trailer" not in html
    assert "🔗 TMDB" not in html
    assert "AI Match" not in html


def test_movie_card_html_shows_match_reasons_and_links():
    card = {
        "title": "Inception",
        "overview": "Dream heist.",
        "rating": 8.8,
        "year": "2010",
        "genres": ["Action", "Science Fiction"],
        "poster": "",
        "runtime": 148,
        "content_type": "movie",
        "trailer_key": "YoHD9XEInc0",
        "tmdb_url": "https://www.themoviedb.org/movie/27205",
        "match_pct": 94,
        "reasons": ["Matches your Sci-Fi preference", "Similar to your picks"],
    }
    html = components.movie_card_html(card)
    assert "2h 28m" in html
    assert "94% AI Match" in html
    assert "Why Priya picked this" in html
    assert "https://www.youtube.com/watch?v=YoHD9XEInc0" in html
    assert "https://www.themoviedb.org/movie/27205" in html


def test_movie_card_html_uses_tv_series_label_when_no_runtime():
    card = {
        "title": "Breaking Bad",
        "overview": "",
        "rating": 9.5,
        "year": "2008",
        "genres": ["Drama"],
        "poster": "",
        "runtime": None,
        "content_type": "tv",
        "trailer_key": "",
        "tmdb_url": "",
        "match_pct": None,
        "reasons": [],
    }
    html = components.movie_card_html(card)
    assert "📺 TV Series" in html
