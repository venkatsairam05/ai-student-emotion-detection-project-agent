"""Tests for the dataset loaders, transforms, and FER2013 parsing."""

from __future__ import annotations

import csv

import pytest

pytest.importorskip("torch")
pytest.importorskip("PIL")
pytest.importorskip("numpy")

import torch

from src.config import EMOTION_LABELS
from src.dataset import (
    EmotionDataset,
    build_transforms,
    class_weights_from_counts,
    create_dataloaders,
    discover_labels,
    parse_fer2013_row,
    summary,
)


def test_discover_labels_orders_canonically(image_folder):
    found = discover_labels(image_folder / "train")
    assert found == ["happy", "sad"]


def test_discover_labels_appends_unknown_alphabetically(tmp_path):
    root = tmp_path / "data"
    for name in ("zzz", "happy", "aaa"):
        (root / name).mkdir(parents=True)
    assert discover_labels(root) == ["happy", "aaa", "zzz"]


def test_discover_labels_on_missing_dir(tmp_path):
    assert discover_labels(tmp_path / "absent") == []


def test_dataset_length_and_labels(image_folder):
    dataset = EmotionDataset(image_folder / "train")
    assert len(dataset) == 6
    assert dataset.labels == ["happy", "sad"]


def test_dataset_returns_image_and_index(image_folder, image_size):
    import torch

    _, eval_tf = build_transforms(image_size)
    dataset = EmotionDataset(image_folder / "train", transform=eval_tf)
    image, target = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, image_size, image_size)
    assert target in (0, 1)


def test_dataset_without_transform_returns_pil(image_folder):
    from PIL import Image

    image, target = EmotionDataset(image_folder / "train")[0]
    assert isinstance(image, Image.Image)
    assert target in (0, 1)


def test_dataset_class_counts(image_folder):
    counts = EmotionDataset(image_folder / "train").class_counts()
    assert counts == {"happy": 3, "sad": 3}


def test_dataset_rejects_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        EmotionDataset(tmp_path / "absent")


def test_dataset_rejects_empty_root(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(ValueError):
        EmotionDataset(root)


def test_transforms_produce_expected_shape():
    import torch
    from PIL import Image

    train_tf, eval_tf = build_transforms(48)
    image = Image.new("RGB", (64, 64), (120, 120, 120))
    assert train_tf(image).shape == (3, 48, 48)
    assert eval_tf(image).shape == (3, 48, 48)


def test_eval_transform_is_deterministic():
    import torch
    from PIL import Image

    _, eval_tf = build_transforms(48)
    image = Image.new("RGB", (48, 48), (10, 200, 30))
    assert torch.allclose(eval_tf(image), eval_tf(image))


def test_dataloaders_share_label_order(image_folder):
    train_loader, val_loader, labels = create_dataloaders(
        train_dir=image_folder / "train",
        val_dir=image_folder / "val",
        batch_size=2,
        num_workers=0,
    )
    assert labels == ["happy", "sad"]
    images, targets = next(iter(train_loader))
    assert images.shape[0] == 2
    assert int(targets.max()) < len(labels)
    assert val_loader is not None


def test_class_weights_favor_minority():
    weights = class_weights_from_counts([100, 10])
    assert weights[1] > weights[0]


def test_class_weights_handle_zero_count():
    weights = class_weights_from_counts([10, 0])
    assert float(weights[1]) == 0.0
    assert bool(torch.isfinite(weights).all())


def test_summary_reports_counts(image_folder):
    info = summary(EmotionDataset(image_folder / "train"))
    assert info["total"] == 6
    assert info["num_classes"] == 2
    assert info["majority"] == 3


# --------------------------------------------------------------------------- #
# FER2013 CSV
# --------------------------------------------------------------------------- #


def test_parse_fer2013_row():
    row = ["3", "0 10 20 30", "Training"]
    parsed = parse_fer2013_row(row)
    assert parsed.emotion == 3
    assert parsed.pixels == (0, 10, 20, 30)
    assert parsed.usage == "Training"


def test_parse_fer2013_row_rejects_short_row():
    with pytest.raises(ValueError):
        parse_fer2013_row(["3", "0 1"])


def test_iter_fer2013_filters_by_usage(tmp_path):
    from src.dataset import iter_fer2013

    path = tmp_path / "fer2013.csv"
    pixels = " ".join(["10"] * (48 * 48))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["emotion", "pixels", "Usage"])
        writer.writerow(["0", pixels, "Training"])
        writer.writerow(["1", pixels, "PublicTest"])
        writer.writerow(["2", pixels, "PrivateTest"])

    training = iter_fer2013(path, usage="Training")
    assert len(training) == 1
    assert training[0][1] == 0
    assert len(iter_fer2013(path)) == 3


def test_iter_fer2013_skips_malformed_pixels(tmp_path):
    from src.dataset import iter_fer2013

    path = tmp_path / "fer2013.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["emotion", "pixels", "Usage"])
        writer.writerow(["0", "1 2 3", "Training"])          # wrong pixel count
        writer.writerow(["99", "0 1 2", "Training"])         # wrong pixel count
        writer.writerow(["0", "5 5 5", "Mystery"])           # unknown usage

    assert iter_fer2013(path) == []


def test_iter_fer2013_missing_file(tmp_path):
    from src.dataset import iter_fer2013

    with pytest.raises(FileNotFoundError):
        iter_fer2013(tmp_path / "absent.csv")


def test_fer2013_labels_match_canonical_order():
    from src.emotions import to_fer2013_pixel_order

    assert to_fer2013_pixel_order() == EMOTION_LABELS
