"""Evaluation: per-class metrics, confusion matrix, and engagement snapshots.

Implemented directly against torch tensors rather than scikit-learn so the
module has no extra dependency and works on a minimal torch install.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from src.config import IMAGE_SIZE, MEAN, STD
from src.emotions import engagement_band, engagement_score

try:  # pragma: no cover - import guard
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


def _require_torch() -> None:
    """Raise a helpful error when PyTorch is unavailable."""

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for evaluation. `pip install torch`.")


def confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
) -> List[List[int]]:
    """Confusion matrix with ``matrix[true][pred]`` counts."""

    _require_torch()
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} vs {len(y_pred)}")
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for true_label, pred_label in zip(y_true, y_pred):
        if not 0 <= true_label < num_classes or not 0 <= pred_label < num_classes:
            continue
        matrix[int(true_label)][int(pred_label)] += 1
    return matrix


def per_class_metrics(matrix: List[List[int]]) -> Dict[str, Dict[str, float]]:
    """Precision, recall, F1, and support for every class.

    Undefined values (zero denominator) are reported as ``0.0`` rather than
    ``nan`` so the values can be charted directly.
    """

    _require_torch()
    num_classes = len(matrix)
    results: Dict[str, Dict[str, float]] = {}
    for index in range(num_classes):
        tp = matrix[index][index]
        fn = sum(matrix[index]) - tp
        fp = sum(row[index] for row in matrix) - tp
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        results[str(index)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }
    return results


def macro_f1(metrics: Dict[str, Dict[str, float]]) -> float:
    """Unweighted mean F1 across classes."""

    if not metrics:
        return 0.0
    return sum(m["f1"] for m in metrics.values()) / len(metrics)


def weighted_f1(
    metrics: Dict[str, Dict[str, float]], matrix: List[List[int]]
) -> float:
    """Support-weighted mean F1 across classes."""

    total = sum(m["support"] for m in metrics.values())
    if total <= 0:
        return 0.0
    return sum(m["f1"] * m["support"] for m in metrics.values()) / total


def top_k_accuracy(y_true: Sequence[int], logits: "torch.Tensor", k: int = 2) -> float:
    """Fraction of samples whose true class is within the top ``k`` logits."""

    _require_torch()
    if len(y_true) == 0:
        return 0.0
    k = max(1, min(k, int(logits.shape[1])))
    top = logits.topk(k, dim=1).indices
    truth = torch.as_tensor(list(y_true), dtype=torch.long)
    hits = (top == truth.unsqueeze(1)).any(dim=1)
    return float(hits.float().mean().item())


def aggregate_metrics(matrix: List[List[int]], top2: float = 0.0) -> Dict[str, object]:
    """Full metric bundle: accuracy, macro/weighted F1, and per-class detail."""

    _require_torch()
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(matrix)))
    metrics = per_class_metrics(matrix)
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "macro_f1": macro_f1(metrics),
        "weighted_f1": weighted_f1(metrics, matrix),
        "top2_accuracy": top2,
        "per_class": metrics,
    }


@torch.no_grad() if TORCH_AVAILABLE else (lambda f: f)
def evaluate_model(
    model: "torch.nn.Module",
    loader: "object",
    labels: Sequence[str],
    device: str = "auto",
    image_size: int = IMAGE_SIZE,
    collect_probabilities: bool = False,
) -> Dict[str, object]:
    """Run ``model`` over ``loader`` and return metrics plus a confusion matrix.

    The confusion matrix rows are real classes, columns are predictions. When
    ``collect_probabilities`` is set, per-sample class distributions are kept
    (used by the engagement snapshots).
    """

    _require_torch()
    from src.checkpoint import resolve_device

    target = resolve_device(device)
    model.to(target)
    model.eval()

    all_true: List[int] = []
    all_pred: List[int] = []
    all_logits: List["torch.Tensor"] = []

    for images, targets in loader:  # type: ignore[misc]
        images = images.to(target)
        logits = model(images)
        if collect_probabilities:
            all_logits.append(logits.detach().cpu())
        all_true.extend(int(t) for t in targets)
        all_pred.extend(int(p) for p in logits.argmax(dim=1).cpu())

    if not all_true:
        return {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "top2_accuracy": 0.0,
            "per_class": {},
            "confusion_matrix": [],
            "labels": list(labels),
        }

    top2 = 0.0
    if all_logits:
        top2 = top_k_accuracy(all_true, torch.cat(all_logits, dim=0), k=2)

    result = aggregate_metrics(confusion_matrix(all_true, all_pred, len(labels)), top2)
    result["confusion_matrix"] = confusion_matrix(all_true, all_pred, len(labels))
    result["labels"] = list(labels)
    result["y_true"] = all_true
    result["y_pred"] = all_pred
    return result


def confusion_dataframe_rows(matrix: List[List[int]], labels: Sequence[str]) -> List[Dict[str, object]]:
    """Rows shaped for ``st.dataframe``: one dict per true class."""

    return [
        {
            "true": label,
            **{f"pred_{labels[j]}": int(matrix[i][j]) for j in range(len(labels))},
            "support": int(sum(matrix[i])),
        }
        for i, label in enumerate(labels)
    ]


def predict_labels(
    model: "torch.nn.Module",
    images: "torch.Tensor",
    labels: Sequence[str],
) -> List[Dict[str, object]]:
    """Per-image label, confidence, and engagement metrics.

    ``images`` is a ``(N, 3, H, W)`` tensor already normalized. Each result is
    ``{"index", "label", "emoji", "confidence", "engagement", "band",
    "probabilities"}``.
    """

    _require_torch()
    if images.numel() == 0:
        return []

    probabilities = model.predict_proba(images)
    top_probs, top_idx = probabilities.max(dim=1)
    results: List[Dict[str, object]] = []
    for row in range(probabilities.shape[0]):
        dist = {
            labels[i]: float(probabilities[row, i]) for i in range(min(len(labels), probabilities.shape[1]))
        }
        label = labels[int(top_idx[row])]
        score = engagement_score(dist)
        results.append(
            {
                "index": int(top_idx[row]),
                "label": label,
                "emoji": {"angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😊",
                          "sad": "😢", "surprise": "😮", "neutral": "😐"}.get(label, "🙂"),
                "confidence": float(top_probs[row]),
                "engagement": score,
                "band": engagement_band(score),
                "probabilities": dist,
            }
        )
    return results


def preprocess_pil(img: "object", image_size: int = IMAGE_SIZE) -> "torch.Tensor":
    """Convert a PIL image to a normalized ``(1, 3, size, size)`` tensor.

    Kept separate from :mod:`src.dataset` so the inference path does not need
    torchvision transforms installed.
    """

    _require_torch()
    resized = img.convert("RGB").resize((image_size, image_size))
    import numpy as np

    array = np.asarray(resized, dtype="float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor(MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean) / std
