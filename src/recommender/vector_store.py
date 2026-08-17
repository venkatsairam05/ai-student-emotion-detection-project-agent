"""FAISS vector store for semantic search over movie embeddings.

Persistence:
    * ``artifacts/faiss.index``  - FAISS index (L2/InnerProduct)
    * ``artifacts/metadata.pkl`` - row metadata aligned with the index
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config.settings import get_settings
from src.utils.logging_config import get_logger

logger = get_logger("vector_store")


def build_index(vectors: np.ndarray) -> Any:
    """Build a FAISS index from an (N, dim) float32 matrix."""
    try:
        import faiss  # type: ignore
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError("faiss-cpu is not installed. Run: pip install faiss-cpu") from exc

    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    vectors = np.ascontiguousarray(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(index: Any, path: Optional[Path] = None) -> Path:
    """Write a FAISS index to disk."""
    target = path or get_settings().faiss_index_path
    target.parent.mkdir(parents=True, exist_ok=True)
    import faiss  # type: ignore

    faiss.write_index(index, str(target))
    logger.info("Saved FAISS index to %s", target)
    return target


def load_index(path: Optional[Path] = None) -> Optional[Any]:
    """Load a FAISS index; returns None when missing/corrupt."""
    source = path or get_settings().faiss_index_path
    if not source.exists():
        return None
    try:
        import faiss  # type: ignore

        index = faiss.read_index(str(source))
        logger.info("Loaded FAISS index (%d vectors) from %s", index.ntotal, source)
        return index
    except Exception as exc:  # pragma: no cover - corrupt index
        logger.warning("Could not load FAISS index: %s", exc)
        return None


def save_metadata(metadata: list[dict], path: Optional[Path] = None) -> Path:
    """Persist metadata aligned with index rows."""
    target = path or get_settings().faiss_metadata_path
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as fh:
        pickle.dump(metadata, fh)
    logger.info("Saved %d metadata rows to %s", len(metadata), target)
    return target


def load_metadata(path: Optional[Path] = None) -> Optional[list[dict]]:
    """Load persisted metadata; None when missing/corrupt."""
    source = path or get_settings().faiss_metadata_path
    if not source.exists():
        return None
    try:
        with source.open("rb") as fh:
            return pickle.load(fh)
    except Exception as exc:  # pragma: no cover - corrupt cache
        logger.warning("Could not load metadata: %s", exc)
        return None


def search(
    index: Any,
    query_vector: np.ndarray,
    k: int = 20,
) -> list[tuple[int, float]]:
    """Return top-k ``[(row_index, similarity), ...]`` for a query vector."""
    if index is None or query_vector is None:
        return []
    query = np.asarray([query_vector], dtype=np.float32)
    similarities, indices = index.search(query, k)
    results = []
    for sim, idx in zip(similarities[0], indices[0]):
        if idx == -1:
            continue
        results.append((int(idx), float(sim)))
    return results


def index_exists(path: Optional[Path] = None) -> bool:
    source = path or get_settings().faiss_index_path
    return source.exists() and get_settings().faiss_metadata_path.exists()
