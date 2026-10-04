"""Input pipeline: FER2013 loading, image-folder datasets, and transforms.

Two input formats are supported:

``fer2013``
    The original ``fer2013.csv`` (pixels as whitespace-separated ints, ``Usage``
    column splitting train/validation/test). This is what
    ``scripts/prepare_data.py`` converts into the image-folder layout below.

``imagefolder``
    A plain directory of class subfolders, e.g. ``data/processed/train/happy/*.png``.
    Anything that yields one image per class works.

Why the transforms are hand-rolled
---------------------------------
``torchvision.transforms`` costs roughly ten seconds just to import on a cold
interpreter here, and that cost lands on every app start and every test run.
These transforms do the same work with NumPy and Pillow, which are already
loaded, so a warm run starts in well under a second. Set ``USE_TORCHVISION=1``
to use torchvision instead when you prefer it.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from src.config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    LABELS_PATH,
    MEAN,
    STD,
    TRAIN_DIR,
    VAL_DIR,
)
from src.emotions import label_to_index

try:  # pragma: no cover - import guard, exercised indirectly
    import torch
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    DataLoader = Dataset = object  # type: ignore
    TORCH_AVAILABLE = False

try:  # pragma: no cover
    import numpy as np
    from PIL import Image, ImageOps

    NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    Image = ImageOps = None  # type: ignore
    NUMPY_AVAILABLE = False

    PIL_AVAILABLE = False
else:
    PIL_AVAILABLE = True


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def _require_torch() -> None:
    """Raise a helpful error when the deep-learning stack is unavailable."""

    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the emotion dataset. Install it with "
            "`pip install torch` (see requirements.txt)."
        )


def _require_pil() -> None:
    """Raise a helpful error when Pillow/NumPy are unavailable."""

    if not PIL_AVAILABLE:
        raise ImportError(
            "Pillow and NumPy are required to load images. `pip install Pillow numpy`."
        )


def use_torchvision() -> bool:
    """Whether to build the transform pipeline from torchvision.

    Off by default because importing torchvision is the single slowest import in
    the project. Opt in with ``USE_TORCHVISION=1``.
    """

    return os.getenv("USE_TORCHVISION", "0").strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #

Transform = Callable[[object], object]


class Compose:
    """Apply a list of callables in order."""

    def __init__(self, steps: Sequence[Transform]) -> None:
        self.steps = list(steps)

    def __call__(self, image: object) -> object:
        for step in self.steps:
            image = step(image)
        return image

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        names = ", ".join(type(step).__name__ for step in self.steps)
        return f"Compose([{names}])"


class Resize:
    """Resize to a fixed square."""

    def __init__(self, size: int) -> None:
        self.size = int(size)

    def __call__(self, image: object):
        return image.resize((self.size, self.size), Image.BILINEAR)


class RandomHorizontalFlip:
    """Mirror the image left-right with probability ``p``."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = float(p)

    def __call__(self, image: object):
        if torch.rand(()) < self.p:
            return ImageOps.mirror(image)
        return image


class RandomRotation:
    """Rotate by a random angle in ``[-degrees, +degrees]``, keeping corners white."""

    def __init__(self, degrees: float = 12.0) -> None:
        self.degrees = float(degrees)

    def __call__(self, image: object):
        angle = float(torch.empty(()).uniform_(-self.degrees, self.degrees))
        return image.rotate(angle, resample=Image.BILINEAR, fillcolor=0)


class ColorJitter:
    """Random brightness and contrast adjustment."""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2) -> None:
        self.brightness = float(brightness)
        self.contrast = float(contrast)

    def __call__(self, image: object):
        array = np.asarray(image, dtype=np.float32)
        if self.brightness > 0:
            factor = 1.0 + float(torch.empty(()).uniform_(-self.brightness, self.brightness))
            array = np.clip(array * factor, 0.0, 255.0)
        if self.contrast > 0:
            factor = 1.0 + float(torch.empty(()).uniform_(-self.contrast, self.contrast))
            mean = array.mean()
            array = np.clip((array - mean) * factor + mean, 0.0, 255.0)
        return Image.fromarray(array.astype(np.uint8))


class ToTensor:
    """Convert a PIL image to a normalized CHW float tensor.

    Values are mapped from ``[0, 255]`` to ``[0, 1]``, then shifted and scaled by
    ``MEAN``/``STD`` - which lands in ``[-1, 1]`` with the default settings.
    """

    def __call__(self, image: object):
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
        return tensor.sub_(MEAN[0]).div_(STD[0])


class RandomErasing:
    """Blank a random rectangle, mimicking ``torchvision.RandomErasing``.

    Operates on the normalized tensor and fills with 0.0, which is the dataset
    mean, so erased regions carry no class signal.
    """

    def __init__(self, p: float = 0.25, scale: Tuple[float, float] = (0.02, 0.12)) -> None:
        self.p = float(p)
        self.scale = scale

    def __call__(self, tensor):
        if float(torch.rand(())) >= self.p:
            return tensor
        _, height, width = tensor.shape
        area = height * width
        for _ in range(10):
            target = float(torch.empty(()).uniform_(self.scale[0], self.scale[1])) * area
            ratio = float(torch.empty(()).uniform_(1 / 3.0, 3.0))
            h = int(round((target * ratio) ** 0.5))
            w = int(round((target / max(ratio, 1e-6)) ** 0.5))
            if h < height and w < width:
                top = int(torch.randint(0, height - h, ()))
                left = int(torch.randint(0, width - w, ()))
                tensor[:, top : top + h, left : left + w] = 0.0
                return tensor
        return tensor


def _build_torchvision_transforms(image_size: int) -> Optional[Tuple[Transform, Transform]]:
    """Build torchvision pipelines, or ``None`` when torchvision is unavailable."""

    try:
        from torchvision import transforms as tv  # type: ignore

        normalize = tv.Normalize(mean=MEAN, std=STD)
        train_tf = tv.Compose(
            [
                tv.Resize((image_size, image_size)),
                tv.RandomHorizontalFlip(p=0.5),
                tv.RandomRotation(degrees=12),
                tv.ColorJitter(brightness=0.2, contrast=0.2),
                tv.ToTensor(),
                tv.RandomErasing(p=0.25, scale=(0.02, 0.12)),
                normalize,
            ]
        )
        eval_tf = tv.Compose(
            [tv.Resize((image_size, image_size)), tv.ToTensor(), normalize]
        )
        return train_tf, eval_tf
    except Exception:
        return None


_TRANSFORM_CACHE: Dict[Tuple[int, bool], Tuple[Transform, Transform]] = {}


def build_transforms(image_size: int = IMAGE_SIZE) -> Tuple[Transform, Transform]:
    """Return ``(train_transform, eval_transform)``, cached per size.

    Training transforms apply label-preserving augmentation; validation and test
    transforms are deterministic so metrics are reproducible.
    """

    _require_pil()
    key = (int(image_size), use_torchvision())
    if key in _TRANSFORM_CACHE:
        return _TRANSFORM_CACHE[key]

    if use_torchvision():
        built = _build_torchvision_transforms(image_size)
        if built is not None:
            _TRANSFORM_CACHE[key] = built
            return built

    train_tf = Compose(
        [
            Resize(image_size),
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=12),
            ColorJitter(brightness=0.2, contrast=0.2),
            ToTensor(),
            RandomErasing(p=0.25),
        ]
    )
    eval_tf = Compose([Resize(image_size), ToTensor()])
    _TRANSFORM_CACHE[key] = (train_tf, eval_tf)
    return _TRANSFORM_CACHE[key]


def clear_transform_cache() -> None:
    """Drop cached pipelines (used by tests that change the input size)."""

    _TRANSFORM_CACHE.clear()


# --------------------------------------------------------------------------- #
# Image-folder dataset
# --------------------------------------------------------------------------- #


class EmotionDataset(Dataset):
    """Grayscale-converted face crops with integer emotion labels.

    Images are converted to RGB so the CNN's first conv sees three channels, and
    pixel values are normalized to ``[-1, 1]``.
    """

    def __init__(
        self,
        root: Path,
        transform: Optional[Transform] = None,
        labels: Sequence[str] = tuple(),
    ) -> None:
        _require_pil()
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")

        self.labels: List[str] = list(labels) or discover_labels(self.root)
        if not self.labels:
            raise ValueError(f"No emotion class folders found under {self.root}")

        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        for class_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if class_dir.name not in self.labels:
                continue
            target = label_to_index(class_dir.name, self.labels)
            for path in sorted(class_dir.glob("*")):
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                    self.samples.append((path, target))

        if not self.samples:
            raise ValueError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):  # type: ignore[no-untyped-def]
        path, target = self.samples[idx]
        with Image.open(path) as img:
            image = ImageOps.exif_transpose(img).convert("L").convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def class_counts(self) -> Dict[str, int]:
        """Number of samples per emotion label, keyed by label name."""

        counts = {label: 0 for label in self.labels}
        for _, target in self.samples:
            counts[self.labels[target]] += 1
        return counts


def discover_labels(root: Path) -> List[str]:
    """Infer the class list from a directory's subfolder names.

    Known labels are ordered canonically (FER2013 order) so class indices stay
    stable across runs; unknown names are appended alphabetically.
    """

    root = Path(root)
    if not root.is_dir():
        return []
    names = {p.name for p in root.iterdir() if p.is_dir()}
    if not names:
        return []

    from src.config import EMOTION_LABELS

    canonical = [label for label in EMOTION_LABELS if label in names]
    extra = sorted(names.difference(canonical))
    return canonical + extra


def load_labels(path: Path = LABELS_PATH) -> List[str]:
    """Read the label order saved next to a trained checkpoint.

    Falls back to the canonical FER2013 order when the file is absent, so an
    untrained checkout still produces usable (if meaningless) predictions.
    """

    from src.config import EMOTION_LABELS

    path = Path(path)
    if path.is_file():
        import json

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            labels = payload.get("labels") if isinstance(payload, dict) else payload
            if isinstance(labels, list) and labels:
                return [str(label) for label in labels]
        except Exception:
            pass
    return list(EMOTION_LABELS)


# --------------------------------------------------------------------------- #
# FER2013 CSV
# --------------------------------------------------------------------------- #

FER2013_COLUMNS = ("emotion", "pixels", "Usage")


@dataclass(frozen=True)
class FerRow:
    """One parsed row of ``fer2013.csv``."""

    emotion: int
    pixels: Tuple[int, ...]
    usage: str


def parse_fer2013_row(row: Sequence[str]) -> FerRow:
    """Convert one CSV row to :class:`FerRow`.

    ``pixels`` is stored as ints (not a 4096-byte string) to keep memory sane
    while parsing a 300 MB file.
    """

    if len(row) < 3:
        raise ValueError(f"Expected at least 3 columns, got {len(row)}")
    emotion = int(row[0].strip())
    pixels = tuple(int(v) for v in row[1].split())
    usage = row[2].strip() or "Unknown"
    return FerRow(emotion=emotion, pixels=pixels, usage=usage)


def iter_fer2013(
    path: Path,
    usage: Optional[str] = None,
    label_count: int = 7,
    image_size: int = IMAGE_SIZE,
) -> List[Tuple["torch.Tensor", int]]:
    """Load ``fer2013.csv`` rows into tensors, optionally filtered by usage.

    ``usage`` accepts ``"Training"``, ``"PublicTest"``, or ``"PrivateTest"``
    (case-insensitive). ``None`` loads every row.
    """

    _require_torch()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"FER2013 CSV not found: {path}")

    wanted = usage.strip().lower() if usage else None
    samples: List[Tuple["torch.Tensor", int]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is not None and tuple(h.strip() for h in header) == FER2013_COLUMNS:
            pass  # header consumed
        elif header is not None:
            handle.seek(0)
            reader = csv.reader(handle)

        for row in reader:
            if not row or not row[0].strip():
                continue
            parsed = parse_fer2013_row(row)
            if wanted is not None and parsed.usage.lower() != wanted:
                continue
            if not 0 <= parsed.emotion < label_count:
                continue
            expected = image_size * image_size
            if len(parsed.pixels) != expected:
                continue
            flat = torch.tensor(parsed.pixels, dtype=torch.float32).view(1, image_size, image_size)
            rgb = flat.repeat(3, 1, 1) / 255.0
            samples.append(((rgb - MEAN[0]) / STD[0], parsed.emotion))
    return samples


def class_weights_from_counts(
    counts: Sequence[int], device: str = "cpu"
) -> "torch.Tensor":
    """Inverse-frequency class weights for imbalanced datasets.

    ``counts[i] == 0`` is skipped rather than producing ``inf``.
    """

    _require_torch()
    total = float(sum(counts))
    weights = []
    for count in counts:
        if count <= 0 or total <= 0:
            weights.append(0.0)
            continue
        weights.append(total / (len(counts) * float(count)))
    return torch.tensor(weights, dtype=torch.float32, device=device)


# --------------------------------------------------------------------------- #
# DataLoaders
# --------------------------------------------------------------------------- #


def create_dataloaders(
    train_dir: Path = TRAIN_DIR,
    val_dir: Path = VAL_DIR,
    image_size: int = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple["DataLoader", "DataLoader", List[str]]:
    """Build train/validation loaders plus the shared label order.

    The label order is taken from ``train_dir`` and applied to ``val_dir`` so
    class indices always line up with the saved checkpoint.
    """

    _require_torch()
    from src.runtime import configure_threads

    # Small batches are faster single-threaded; see src.runtime.recommended_threads.
    configure_threads(batch_size, image_size)

    train_tf, eval_tf = build_transforms(image_size)
    labels = discover_labels(Path(train_dir))
    if not labels:
        raise ValueError(f"No class folders found in {train_dir}")

    train_ds = EmotionDataset(train_dir, transform=train_tf, labels=labels)
    val_ds = EmotionDataset(val_dir, transform=eval_tf, labels=labels)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        # BatchNorm1d in the head cannot learn from a single sample, and a
        # trailing batch of one is the classic way to crash a training run.
        # Only drop when doing so still leaves at least one full batch.
        drop_last=len(train_ds) > batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, labels


def summary(dataset: "EmotionDataset") -> Dict[str, object]:
    """Small dict describing a dataset, used by CLI scripts and the UI."""

    counts = dataset.class_counts()
    total = sum(counts.values())
    return {
        "root": str(dataset.root),
        "total": total,
        "num_classes": len(dataset.labels),
        "labels": dataset.labels,
        "counts": counts,
        "minority": min(counts.values()) if counts else 0,
        "majority": max(counts.values()) if counts else 0,
    }
