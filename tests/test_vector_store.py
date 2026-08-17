"""Tests for FAISS vector store + embeddings persistence."""

from __future__ import annotations

import numpy as np

from src.recommender import vector_store
from tests.conftest import pseudo_embedding


def test_build_and_search(tmp_path):
    texts = [
        "sci-fi space thriller",
        "romantic comedy in new york",
        "horror movie haunted house",
        "documentary about oceans",
    ]
    vectors = np.stack([pseudo_embedding(t) for t in texts])
    index = vector_store.build_index(vectors)
    assert index.ntotal == 4

    results = vector_store.search(index, pseudo_embedding("space sci-fi"), k=2)
    assert len(results) == 2
    # The closest item to "space sci-fi" is texts[0].
    assert results[0][0] == 0


def test_save_load_index(tmp_path):
    vectors = np.stack([pseudo_embedding(f"movie {i}") for i in range(5)])
    index = vector_store.build_index(vectors)
    path = tmp_path / "faiss.index"
    vector_store.save_index(index, path)
    loaded = vector_store.load_index(path)
    assert loaded is not None
    assert loaded.ntotal == 5


def test_load_missing_index_returns_none(tmp_path):
    assert vector_store.load_index(tmp_path / "missing.index") is None


def test_metadata_roundtrip(tmp_path):
    metadata = [{"id": 1, "title": "A"}, {"id": 2, "title": "B"}]
    path = tmp_path / "meta.pkl"
    vector_store.save_metadata(metadata, path)
    loaded = vector_store.load_metadata(path)
    assert loaded == metadata


def test_search_returns_empty_for_none():
    assert vector_store.search(None, np.zeros(8, dtype=np.float32), k=5) == []
