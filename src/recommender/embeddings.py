"""Sentence-transformer embeddings with batch generation and persistence.

Embeddings are generated once by ``scripts/build_index.py`` and cached to
``artifacts/embeddings.npy`` so the Streamlit app never regenerates them at
startup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import get_settings
from src.utils.logging_config import get_logger

logger = get_logger("embeddings")

_model = None
_model_name_loaded: str = ""


def load_model(model_name: Optional[str] = None):
    """Load (and cache) the SentenceTransformer model.

    Falls back to ``None`` when the model cannot be downloaded/loaded so the
    rest of the app can degrade gracefully.
    """
    global _model, _model_name_loaded  # noqa: PLW0603
    name = model_name or get_settings().embedding_model_name
    if _model is not None and _model_name_loaded == name:
        return _model
    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(name)
        _model_name_loaded = name
        logger.info("Embedding model loaded: %s", name)
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.error("Failed to load embedding model %s: %s", name, exc)
        _model = None
        _model_name_loaded = ""
    return _model


def generate_embedding(text: str, model=None) -> Optional[np.ndarray]:
    """Embed a single text string; returns a float32 vector or None."""
    model = model or load_model()
    if model is None:
        return None
    vector = model.encode([text], normalize_embeddings=True)[0]
    return np.asarray(vector, dtype=np.float32)


def generate_embeddings(
    texts: list[str],
    model=None,
    batch_size: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Embed a list of texts in batches.

    Returns an (N, dim) float32 matrix, or None if the model is unavailable.
    """
    if not texts:
        return None
    model = model or load_model()
    if model is None:
        return None
    batch = batch_size or get_settings().embedding_batch_size
    vectors = model.encode(
        texts,
        batch_size=batch,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def save_embeddings(vectors: np.ndarray, path: Optional[Path] = None) -> Path:
    """Persist embeddings to disk (default ``artifacts/embeddings.npy``)."""
    target = path or get_settings().embeddings_path
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, vectors)
    logger.info("Saved %d embeddings to %s", len(vectors), target)
    return target


def load_embeddings(path: Optional[Path] = None) -> Optional[np.ndarray]:
    """Load embeddings from disk; returns None when missing/invalid."""
    source = path or get_settings().embeddings_path
    if not source.exists():
        return None
    try:
        return np.load(source).astype(np.float32)
    except Exception as exc:  # pragma: no cover - corrupt cache
        logger.warning("Could not load embeddings from %s: %s", source, exc)
        return None
