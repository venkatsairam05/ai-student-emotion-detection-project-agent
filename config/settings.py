"""Configuration management for CineMatch AI.

All settings are read from environment variables (via a ``.env`` file when
present) with sensible local defaults so the project can start up without any
secrets configured. API keys are never hardcoded in source code.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Project root = two directories up from this file (config/settings.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root (no-op if the file is missing).
load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    """Runtime configuration container backed by environment variables."""

    @staticmethod
    def _real_key(value: str) -> str:
        """Return ``value`` unless it is empty or an unfilled template placeholder."""
        value = (value or "").strip()
        if not value or value.startswith("YOUR_") or "REPLACE" in value.upper():
            return ""
        return value

    def __init__(self) -> None:
        # --- API keys ----------------------------------------------------
        self.anthropic_api_key: str = self._real_key(os.getenv("ANTHROPIC_API_KEY", ""))
        self.tmdb_api_key: str = self._real_key(os.getenv("TMDB_API_KEY", ""))
        self.anthropic_model: str = os.getenv(
            "ANTHROPIC_MODEL", "claude-3-5-haiku-latest"
        )

        # --- Paths -------------------------------------------------------
        self.data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
        self.data_processed_dir: Path = PROJECT_ROOT / "data" / "processed"
        self.artifacts_dir: Path = PROJECT_ROOT / "artifacts"
        self.raw_dataset_path: Path = self.data_raw_dir / "movies_metadata.csv"
        self.processed_dataset_path: Path = (
            self.data_processed_dir / "processed_movies.csv"
        )
        self.embeddings_path: Path = self.artifacts_dir / "embeddings.npy"
        self.faiss_index_path: Path = self.artifacts_dir / "faiss.index"
        self.faiss_metadata_path: Path = self.artifacts_dir / "metadata.pkl"

        # --- Embedding model --------------------------------------------
        self.embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_dim: int = 384
        self.embedding_batch_size: int = 32

        # --- Recommender defaults ---------------------------------------
        self.top_k: int = 5
        self.semantic_candidates: int = 50
        # Default ranking weights (configurable per call).
        self.ranking_weights: dict[str, float] = {
            "semantic_similarity": 0.50,
            "preference_match": 0.20,
            "rating": 0.15,
            "popularity": 0.10,
            "freshness": 0.05,
        }

        # --- TMDB --------------------------------------------------------
        self.tmdb_base_url: str = "https://api.themoviedb.org/3"
        self.tmdb_image_base_url: str = os.getenv(
            "TMDB_IMAGE_BASE_URL", "https://image.tmdb.org/t/p/w500"
        )
        self.tmdb_timeout_seconds: float = 8.0
        self.tmdb_retries: int = 2
        self.tmdb_rate_limit_sleep: float = 1.0
        # How many top search results get a lightweight live enrich
        # (runtime / trailer / backdrop) on first search. Kept small so a
        # single query stays well within TMDB's rate limits.
        self.tmdb_search_enrich_limit: int = int(
            os.getenv("TMDB_SEARCH_ENRICH_LIMIT", "8")
        )

        # --- Claude ------------------------------------------------------
        self.claude_timeout_seconds: float = 30.0
        self.claude_max_tokens: int = 1024

        # --- Voice assistant (Priya) ------------------------------------
        self.voice_assistant_name: str = os.getenv("VOICE_ASSISTANT_NAME", "Priya")
        self.voice_enabled_default: bool = os.getenv("VOICE_ENABLED", "true").lower() in ("1", "true", "yes")
        # Primary TTS voice (edge-tts) - a natural female voice.
        self.tts_voice: str = os.getenv("TTS_VOICE", "en-IN-PriyaNeural")
        self.tts_voice_alt: str = os.getenv("TTS_VOICE_ALT", "en-US-JennyNeural")
        # Offline TTS fallback (pyttsx3 / SAPI5) female voice keywords.
        self.tts_fallback_voice_keywords: list[str] = [
            "priya", "jenny", "zira", "hazel", "aria", "female", "woman", "susan",
        ]
        self.stt_language: str = os.getenv("STT_LANGUAGE", "en-US")
        self.stt_google_api: str = os.getenv("STT_GOOGLE_API", "google")
        self.tts_speed: float = float(os.getenv("TTS_SPEED", "1.0"))
        # Seconds before online TTS gives up (keeps replies snappy offline).
        self.tts_timeout: float = float(os.getenv("TTS_TIMEOUT", "12"))

        # --- Runtime -----------------------------------------------------
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")

    @property
    def has_anthropic_key(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_tmdb_key(self) -> bool:
        return bool(self.tmdb_api_key)

    def ensure_dirs(self) -> None:
        """Create every directory the app depends on if missing."""
        for path in (
            self.data_raw_dir,
            self.data_processed_dir,
            self.artifacts_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, process-wide :class:`Settings` instance."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
