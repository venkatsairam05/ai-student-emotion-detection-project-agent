"""Shared pytest fixtures.

Builds a tiny in-memory dataset + FAISS index using deterministic
pseudo-embeddings so tests run quickly and without network access.
"""

from __future__ import annotations

import hashlib
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.recommender import embeddings, vector_store
from src.recommender.recommender import Recommender

DIM = 384

_WORD_VECTORS: dict[str, np.ndarray] = {}


def _word_vector(word: str) -> np.ndarray:
    if word not in _WORD_VECTORS:
        seed = int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:4], "little")
        rng = np.random.RandomState(seed)
        _WORD_VECTORS[word] = rng.normal(size=DIM).astype(np.float32)
    return _WORD_VECTORS[word]


def pseudo_embedding(text: str) -> np.ndarray:
    """Deterministic bag-of-words embedding (no model or network needed).

    Vectors for texts sharing words are similar, which is all the tests
    require for semantic search to behave sensibly.
    """
    words = str(text).lower().split()
    if not words:
        return np.zeros(DIM, dtype=np.float32)
    vec = sum(_word_vector(w) for w in words) / float(len(words))
    vec = vec / (np.linalg.norm(vec) + 1e-9)
    return vec.astype(np.float32)


def _seed_rows() -> list[dict]:
    return [
        dict(id=1, title="Inception", overview="Dream heist sci-fi thriller",
             genres="Action|Science Fiction|Thriller",
             keywords="heist|dream|time travel",
             cast="Leonardo DiCaprio|Joseph Gordon-Levitt",
             director="Christopher Nolan", rating=8.8, popularity=95.0,
             release_date="2010-07-16", release_year=2010, runtime=148,
             language="en", content_type="movie",
             combined_text="inception dream heist sci-fi thriller leonardo dicaprio christopher nolan"),
        dict(id=2, title="Interstellar", overview="Space survival epic",
             genres="Adventure|Drama|Science Fiction",
             keywords="space|time travel|wormhole",
             cast="Matthew McConaughey|Anne Hathaway",
             director="Christopher Nolan", rating=8.7, popularity=86.0,
             release_date="2014-11-07", release_year=2014, runtime=169,
             language="en", content_type="movie",
             combined_text="interstellar space survival epic matthew mcconaughey anne hathaway christopher nolan"),
        dict(id=3, title="The Conjuring", overview="Haunted house horror",
             genres="Horror|Mystery",
             keywords="haunted house|ghost",
             cast="Vera Farmiga|Patrick Wilson",
             director="James Wan", rating=7.5, popularity=60.0,
             release_date="2013-07-19", release_year=2013, runtime=112,
             language="en", content_type="movie",
             combined_text="the conjuring haunted house horror vera farmiga patrick wilson james wan"),
        dict(id=4, title="Superbad", overview="High school comedy",
             genres="Comedy",
             keywords="high school|party|teen",
             cast="Jonah Hill|Michael Cera",
             director="Greg Mottola", rating=7.6, popularity=55.0,
             release_date="2007-08-17", release_year=2007, runtime=113,
             language="en", content_type="movie",
             combined_text="superbad high school comedy party teen jonah hill michael cera"),
        dict(id=5, title="Squid Game", overview="Survival game drama series",
             genres="Action|Drama|Thriller",
             keywords="survival|games|dystopia",
             cast="Lee Jung-jae|Park Hae-soo",
             director="Hwang Dong-hyuk", rating=8.0, popularity=90.0,
             release_date="2021-09-17", release_year=2021, runtime=55,
             language="ko", content_type="tv",
             combined_text="squid game survival game drama series korean dystopia"),
        dict(id=6, title="The Office", overview="Workplace mockumentary comedy",
             genres="Comedy",
             keywords="workplace|mockumentary|sitcom",
             cast="Steve Carell|John Krasinski",
             director="Greg Daniels", rating=8.9, popularity=85.0,
             release_date="2005-03-24", release_year=2005, runtime=22,
             language="en", content_type="tv",
             combined_text="the office workplace mockumentary comedy sitcom steve carell"),
        dict(id=7, title="Memento", overview="Memory loss mystery",
             genres="Mystery|Thriller",
             keywords="memory|amnesia",
             cast="Guy Pearce|Carrie-Anne Moss",
             director="Christopher Nolan", rating=8.4, popularity=44.0,
             release_date="2000-10-11", release_year=2000, runtime=113,
             language="en", content_type="movie",
             combined_text="memento memory loss mystery thriller guy pearce christopher nolan"),
        dict(id=8, title="The Dark Knight", overview="Superhero crime drama",
             genres="Action|Crime|Drama|Thriller",
             keywords="superhero|villain|chaos",
             cast="Christian Bale|Heath Ledger",
             director="Christopher Nolan", rating=9.0, popularity=98.0,
             release_date="2008-07-18", release_year=2008, runtime=152,
             language="en", content_type="movie",
             combined_text="the dark knight superhero crime drama heath ledger christian bale christopher nolan"),
        dict(id=9, title="Parasite", overview="Class satire thriller",
             genres="Comedy|Drama|Thriller",
             keywords="class|satire|twist",
             cast="Song Kang-ho|Lee Sun-kyun",
             director="Bong Joon-ho", rating=8.5, popularity=83.0,
             release_date="2019-05-30", release_year=2019, runtime=132,
             language="ko", content_type="movie",
             combined_text="parasite class satire thriller korean bong joon-ho"),
        dict(id=10, title="La La Land", overview="Musical romance drama",
             genres="Comedy|Drama|Music|Romance",
             keywords="musical|romance|dream",
             cast="Ryan Gosling|Emma Stone",
             director="Damien Chazelle", rating=8.0, popularity=66.0,
             release_date="2016-12-09", release_year=2016, runtime=128,
             language="en", content_type="movie",
             combined_text="la la land musical romance drama ryan gosling emma stone"),
    ]


@pytest.fixture(scope="session")
def movies_df() -> pd.DataFrame:
    return pd.DataFrame(_seed_rows())


@pytest.fixture(scope="session")
def index_and_metadata(movies_df):
    """Build a FAISS index + metadata list for the seed dataframe."""
    rows = movies_df.to_dict(orient="records")
    vectors = np.stack([pseudo_embedding(row["combined_text"]) for row in rows])
    index = vector_store.build_index(vectors)
    return index, rows


@pytest.fixture(scope="session")
def small_recommender(movies_df, index_and_metadata, monkeypatch_session):
    index, metadata = index_and_metadata
    recommender = Recommender(index=index, metadata=metadata, movies=movies_df)
    return recommender


class _MonkeypatchSession:
    """Minimal session-scoped monkeypatch shim."""

    def __init__(self) -> None:
        self._patches = []

    def setattr(self, target, name, value):
        import builtins

        original = getattr(target, name)
        setattr(target, name, value)
        self._patches.append((target, name, original))

    def undo(self):
        for target, name, original in reversed(self._patches):
            setattr(target, name, original)


@pytest.fixture(scope="session")
def monkeypatch_session():
    patch = _MonkeypatchSession()
    patch.setattr(embeddings, "generate_embedding", pseudo_embedding)
    yield patch
    patch.undo()


@pytest.fixture(scope="function")
def mock_tmdb_get():
    """Patch ``requests.get`` inside the TMDB client."""
    from src.services import tmdb_client

    with pytest.MonkeyPatch().context() as mp:
        fake = mock.Mock()
        mp.setattr(tmdb_client.requests, "get", fake)
        yield fake
