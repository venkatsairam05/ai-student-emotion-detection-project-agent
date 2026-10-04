"""The emotion CNN: a compact VGG-style network built from scratch in PyTorch.

Design notes
------------
*Input*  48x48x3 RGB faces (FER2013 native resolution), normalized to [-1, 1].
*Blocks*  Five convolutional stages, each doubling spatial resolution until the
feature map is 3x3. BatchNorm + ReLU after every conv.
*Head*    Global average pooling -> 256-unit dense -> dropout -> logits.

The network is deliberately small (~300K parameters). FER2013 images are tiny
and heavily augmented, so a bigger model overfits within a few epochs; widening
and deepening both hurt more than they help on this dataset.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.config import IMAGE_SIZE, ModelConfig

try:  # pragma: no cover - import guard
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    TORCH_AVAILABLE = False


def _require_torch() -> None:
    """Raise a helpful error when PyTorch is unavailable."""

    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the emotion model. Install it with "
            "`pip install torch` (see requirements.txt)."
        )


def _conv_block(in_channels: int, out_channels: int, pool: bool = True) -> "nn.Sequential":
    """Conv-BN-ReLU, optionally followed by 2x2 max pooling."""

    layers: List[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class EmotionCNN(nn.Module):
    """Seven-class emotion classifier over 48x48 face crops."""

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        _require_torch()
        super().__init__()
        self.config = config or ModelConfig()
        w = self.config.width

        self.features = nn.Sequential(
            _conv_block(3, w),          # 48 -> 24
            _conv_block(w, w),          # 24 -> 12
            _conv_block(w, w * 2),      # 12 -> 6
            _conv_block(w * 2, w * 2),  # 6  -> 3
            _conv_block(w * 2, w * 4),  # 3  -> 1 (pooled)
        )
        self.spatial_dropout = nn.Dropout2d(p=self.config.spatial_dropout)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w * 4, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=self.config.dropout),
            nn.Linear(256, self.config.num_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Run the forward pass, returning raw logits of shape ``(N, C)``."""

        features = self.features(x)
        features = self.spatial_dropout(features)
        return self.head(features)

    @torch.no_grad() if TORCH_AVAILABLE else (lambda f: f)
    def predict_proba(self, x: "torch.Tensor") -> "torch.Tensor":
        """Softmax probabilities for ``x``; used by inference and evaluation."""

        self.eval()
        return F.softmax(self.forward(x), dim=1)

    def num_parameters(self) -> int:
        """Total trainable parameter count."""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TinyEmotionCNN(EmotionCNN):
    """A deliberately small variant used by the test suite.

    Exercises the same code paths as :class:`EmotionCNN` at a fraction of the
    compute, so tests stay fast without monkeypatching the architecture.
    """

    def __init__(self, num_classes: int = 7, image_size: int = IMAGE_SIZE) -> None:
        _require_torch()
        cfg = ModelConfig(num_classes=num_classes, width=4, dropout=0.0, spatial_dropout=0.0)
        super().__init__(cfg)
        self.image_size = image_size


def build_model(
    config: Optional[ModelConfig] = None,
    tiny: bool = False,
) -> "EmotionCNN":
    """Factory for the model, mirroring the trainer's CLI ``--tiny`` flag."""

    if tiny:
        return TinyEmotionCNN(num_classes=(config.num_classes if config else 7))
    return EmotionCNN(config)


def describe_architecture(model: "EmotionCNN") -> Dict[str, object]:
    """Summary of a model instance for the UI's model-info panel."""

    return {
        "class": type(model).__name__,
        "parameters": model.num_parameters(),
        "num_classes": model.config.num_classes,
        "width": model.config.width,
        "dropout": model.config.dropout,
        "spatial_dropout": model.config.spatial_dropout,
        "input_size": getattr(model, "image_size", IMAGE_SIZE),
    }


def layer_summary(model: "EmotionCNN", max_lines: int = 60) -> List[str]:
    """One line per conv/linear layer, for ``print(model)``-style output."""

    rows: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            rows.append(
                f"Conv2d({name}): {module.in_channels}->{module.out_channels} "
                f"k={module.kernel_size[0]} p={module.padding[0]}"
            )
        elif isinstance(module, nn.Linear):
            rows.append(f"Linear({name}): {module.in_features}->{module.out_features}")
    return rows[:max_lines]
