"""Training loop: epochs, early stopping, metrics, and history logging.

Deliberately dependency-light: a plain function, no Lightning, no Accelerate.
The loop is short enough to read in one sitting, which matters more for a
project meant to be studied than framework ergonomics would.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from src.config import TrainConfig, ensure_dirs

try:  # pragma: no cover - import guard
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    DataLoader = object  # type: ignore
    TORCH_AVAILABLE = False


def _require_torch() -> None:
    """Raise a helpful error when PyTorch is unavailable."""

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training. `pip install torch`.")


@dataclass
class EpochMetrics:
    """Per-epoch train/validation summary."""

    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: float
    seconds: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        return {
            "epoch": self.epoch,
            "train_loss": round(self.train_loss, 6),
            "val_loss": round(self.val_loss, 6),
            "train_accuracy": round(self.train_accuracy, 6),
            "val_accuracy": round(self.val_accuracy, 6),
            "learning_rate": self.learning_rate,
            "seconds": round(self.seconds, 3),
            **self.extra,
        }


class EarlyStopping:
    """Stop when validation loss stops improving.

    Tracks the best loss and how many epochs have passed without an
    improvement; ``should_stop`` flips once ``patience`` is exhausted. The best
    weights are kept by the caller via :meth:`state_dict`.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = max(0, int(patience))
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.counter = 0

    def step(self, val_loss: float, epoch: int) -> bool:
        """Record ``val_loss``; return True when training should stop."""

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return False
        self.counter += 1
        return self.patience > 0 and self.counter >= self.patience

    @property
    def should_stop(self) -> bool:
        """True once patience has been exhausted."""

        return self.patience > 0 and self.counter >= self.patience


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch for reproducible runs."""

    _require_torch()
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _class_weights_from_loader(
    loader: "DataLoader", num_classes: int, device: "torch.device"
) -> "torch.Tensor":
    """Inverse-frequency weights computed from the loader's dataset."""

    dataset = getattr(loader, "dataset", None)
    if dataset is None or not hasattr(dataset, "class_counts"):
        return torch.ones(num_classes, device=device)
    from src.config import EMOTION_LABELS

    counts_map = dataset.class_counts()
    counts = [float(counts_map.get(label, 0)) for label in dataset.labels or EMOTION_LABELS]
    total = sum(counts)
    if total <= 0:
        return torch.ones(num_classes, device=device)
    return torch.tensor(
        [total / (num_classes * c) if c > 0 else 0.0 for c in counts],
        dtype=torch.float32,
        device=device,
    )


def run_one_epoch(
    model: "torch.nn.Module",
    loader: "DataLoader",
    criterion: "nn.Module",
    device: "torch.device",
    optimizer: Optional["torch.optim.Optimizer"] = None,
    class_weights: Optional["torch.Tensor"] = None,
) -> Tuple[float, float]:
    """Run one pass over ``loader``.

    With ``optimizer=None`` this is an evaluation pass (no grad, ``eval`` mode).
    Otherwise it trains for one epoch and returns mean ``(loss, accuracy)``.
    """

    _require_torch()
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    seen = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = F.cross_entropy(logits, targets, weight=class_weights)

        batch = targets.size(0)
        total_loss += float(loss.item()) * batch
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        seen += batch

    if seen == 0:
        return 0.0, 0.0
    return total_loss / seen, correct / seen


def train(
    model: "torch.nn.Module",
    train_loader: "DataLoader",
    val_loader: "DataLoader",
    config: Optional[TrainConfig] = None,
    device: str = "auto",
    log_every: int = 1,
    progress_callback: Optional[Callable[[str, Dict[str, float]], None]] = None,
) -> Tuple["torch.nn.Module", List[EpochMetrics]]:
    """Train ``model``, restoring the best-validation weights at the end.

    ``progress_callback(message, metrics)`` is invoked after each epoch so the
    Streamlit training page can stream updates without knowing about torch.
    """

    _require_torch()
    from src.checkpoint import resolve_device

    config = config or TrainConfig()
    target = resolve_device(device or config.device)
    set_seed(config.seed)

    model.to(target)
    num_classes = int(getattr(getattr(model, "config", None), "num_classes", 7))

    weights = (
        _class_weights_from_loader(train_loader, num_classes, target)
        if config.use_class_weights
        else None
    )
    criterion = nn.CrossEntropyLoss(
        weight=weights, label_smoothing=config.label_smoothing
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, config.patience // 2)
    )
    stopper = EarlyStopping(patience=config.patience)

    history: List[EpochMetrics] = []
    best_state: Optional[Dict[str, "torch.Tensor"]] = None
    best_val_accuracy = -1.0

    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        lr_before = float(optimizer.param_groups[0]["lr"])

        train_loss, train_acc = run_one_epoch(
            model, train_loader, criterion, target, optimizer=optimizer
        )
        val_loss, val_acc = run_one_epoch(
            model, val_loader, criterion, target, class_weights=weights
        )
        scheduler.step(val_loss)

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_acc,
            val_accuracy=val_acc,
            learning_rate=lr_before,
            seconds=time.perf_counter() - started,
        )
        history.append(metrics)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if progress_callback is not None:
            progress_callback("epoch", metrics.to_dict())
        if epoch % max(1, log_every) == 0:
            print(
                f"epoch {epoch:>3}/{config.epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                f"lr {lr_before:.2e} | {metrics.seconds:.1f}s"
            )

        if stopper.step(val_loss, epoch):
            if progress_callback is not None:
                progress_callback("early_stop", {"epoch": float(epoch)})
            print(f"early stopping at epoch {epoch} (best val loss {stopper.best_loss:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(target)
    model.eval()
    return model, history


def save_history(
    history: Sequence[EpochMetrics], path: Path
) -> Path:
    """Write the per-epoch history to JSON for the UI's charts."""

    ensure_dirs()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([m.to_dict() for m in history], indent=2), encoding="utf-8"
    )
    return path


def load_history(path: Path) -> List[Dict[str, float]]:
    """Read a saved history file; returns ``[]`` when missing or corrupt."""

    path = Path(path)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return payload if isinstance(payload, list) else []
