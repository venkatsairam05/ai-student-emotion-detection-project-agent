"""Model persistence: save, load, and fingerprint emotion checkpoints.

A checkpoint is a single ``.pt`` file holding weights, the label order, and the
architecture config. Keeping the label order *inside* the checkpoint is what
prevents the classic "loaded fine but predictions are scrambled" bug: the class
index -> label mapping travels with the weights.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.config import CHECKPOINT_PATH, LABELS_PATH, ModelConfig, ensure_dirs

try:  # pragma: no cover - import guard
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


def _require_torch() -> None:
    """Raise a helpful error when PyTorch is unavailable."""

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to load checkpoints. `pip install torch`.")


def resolve_device(device: str = "auto") -> "torch.device":
    """Map ``"auto"``/``"cpu"``/``"cuda"``/``"mps"`` to a concrete device."""

    _require_torch()
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    model: "torch.nn.Module",
    labels: Sequence[str],
    path: Path = CHECKPOINT_PATH,
    extra: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, float]]] = None,
) -> Path:
    """Persist weights plus metadata, and mirror the labels to JSON.

    Returns the checkpoint path so callers can log it.
    """

    _require_torch()
    ensure_dirs()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "state_dict": model.state_dict(),
        "labels": list(labels),
        "model_config": asdict(getattr(model, "config", ModelConfig())),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "format_version": 1,
    }
    if extra:
        payload["extra"] = dict(extra)
    if history:
        payload["history"] = list(history)

    torch.save(payload, path)
    LABELS_PATH.write_text(
        json.dumps({"labels": list(labels)}, indent=2), encoding="utf-8"
    )
    return path


def load_checkpoint(
    path: Path = CHECKPOINT_PATH,
    device: str = "auto",
) -> "tuple[torch.nn.Module, List[str], Dict[str, Any]]":
    """Load a checkpoint and rebuild the matching model.

    Returns ``(model, labels, metadata)``. ``model`` is already in ``eval`` mode
    and moved to the resolved device. Raises ``FileNotFoundError`` when the
    checkpoint is absent so callers can show a friendly "train first" message.
    """

    _require_torch()
    from src.model import EmotionCNN

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No trained checkpoint at {path}. Train one with `python scripts/train.py`."
        )

    target = resolve_device(device)
    payload = torch.load(path, map_location=target, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"{path} is not a valid emotion checkpoint")

    labels: List[str] = list(payload.get("labels") or [])
    model_config = ModelConfig(**payload["model_config"]) if "model_config" in payload else ModelConfig()
    if labels:
        model_config.num_classes = len(labels)

    model = EmotionCNN(model_config)
    model.load_state_dict(payload["state_dict"])
    model.to(target)
    model.eval()

    metadata: Dict[str, Any] = {
        "path": str(path),
        "saved_at": payload.get("saved_at"),
        "num_parameters": payload.get("num_parameters"),
        "format_version": payload.get("format_version", 1),
        "extra": payload.get("extra", {}),
        "history": payload.get("history", []),
        "device": str(target),
    }
    return model, labels, metadata


def checkpoint_exists(path: Path = CHECKPOINT_PATH) -> bool:
    """True when a trained checkpoint is available."""

    return Path(path).is_file()


def checkpoint_metadata(path: Path = CHECKPOINT_PATH) -> Dict[str, Any]:
    """Cheap metadata read without instantiating the model.

    Uses ``torch.load`` with ``map_location="cpu"``; falls back to an empty
    dict when torch or the file is unavailable.
    """

    path = Path(path)
    if not TORCH_AVAILABLE or not path.is_file():
        return {}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        "path": str(path),
        "labels": list(payload.get("labels") or []),
        "saved_at": payload.get("saved_at"),
        "num_parameters": payload.get("num_parameters"),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
    }
