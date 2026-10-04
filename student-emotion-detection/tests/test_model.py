"""Tests for the CNN architecture and checkpoint persistence."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.model import EmotionCNN, TinyEmotionCNN, build_model, describe_architecture, layer_summary


def single_batch() -> "torch.Tensor":
    """A single 48x48x3 batch of zeros for shape assertions."""

    return torch.zeros(1, 3, 48, 48)


def test_forward_shape(tiny_model, batch_tensor, labels):
    output = tiny_model(batch_tensor)
    assert output.shape == (batch_tensor.shape[0], len(labels))


def test_predict_proba_sums_to_one(tiny_model, batch_tensor):
    probabilities = tiny_model.predict_proba(batch_tensor)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(batch_tensor.shape[0]), atol=1e-5)
    assert bool((probabilities >= 0).all())


def test_predict_proba_argmax_matches_forward(tiny_model, batch_tensor):
    logits = tiny_model(batch_tensor)
    probabilities = tiny_model.predict_proba(batch_tensor)
    assert torch.equal(logits.argmax(dim=1), probabilities.argmax(dim=1))


def test_model_has_trainable_parameters(tiny_model):
    assert tiny_model.num_parameters() > 0


def test_build_model_default_is_larger_than_tiny():
    assert build_model().num_parameters() > build_model(tiny=True).num_parameters()


def test_emotion_cnn_uses_batchnorm_and_pooling():
    kinds = {type(module).__name__ for module in EmotionCNN().modules()}
    assert {"BatchNorm2d", "MaxPool2d", "Dropout2d"}.issubset(kinds)


def test_eval_mode_is_deterministic(tiny_model, batch_tensor):
    tiny_model.eval()
    with torch.no_grad():
        assert torch.allclose(tiny_model(batch_tensor), tiny_model(batch_tensor))


def test_describe_architecture_fields():
    info = describe_architecture(build_model(tiny=True))
    assert set(info) >= {"class", "parameters", "num_classes", "width"}
    assert info["num_classes"] == 7


def test_layer_summary_mentions_conv_and_linear():
    rows = layer_summary(build_model(tiny=True))
    assert any(row.startswith("Conv2d") for row in rows)
    assert any(row.startswith("Linear") for row in rows)


def test_tiny_model_respects_num_classes():
    model = TinyEmotionCNN(num_classes=3)
    model.eval()
    with torch.no_grad():
        assert model(single_batch()).shape[-1] == 3


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #


def test_checkpoint_roundtrip(tmp_path, labels):
    from src.checkpoint import checkpoint_exists, load_checkpoint, save_checkpoint

    path = tmp_path / "model.pt"
    model = build_model(tiny=True)
    model.eval()
    with torch.no_grad():
        before = model.predict_proba(single_batch())

    save_checkpoint(model, labels, path=path)
    assert checkpoint_exists(path)

    loaded, loaded_labels, metadata = load_checkpoint(path, device="cpu")
    assert loaded_labels == labels
    assert metadata["format_version"] == 1

    with torch.no_grad():
        after = loaded.predict_proba(single_batch())
    assert torch.allclose(before, after, atol=1e-6)


def test_checkpoint_preserves_label_order(tmp_path, labels):
    from src.checkpoint import load_checkpoint, save_checkpoint

    shuffled = list(reversed(labels))
    save_checkpoint(build_model(tiny=True), shuffled, path=tmp_path / "m.pt")
    _, loaded_labels, _ = load_checkpoint(tmp_path / "m.pt", device="cpu")
    assert loaded_labels == shuffled


def test_load_missing_checkpoint_raises(tmp_path):
    from src.checkpoint import load_checkpoint

    with pytest.raises(FileNotFoundError, match="No trained checkpoint"):
        load_checkpoint(tmp_path / "nope.pt", device="cpu")


def test_load_invalid_checkpoint_raises(tmp_path):
    from src.checkpoint import load_checkpoint

    torch.save({"not_a_model": True}, tmp_path / "bad.pt")
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "bad.pt", device="cpu")


def test_save_writes_labels_json(tmp_path, labels, monkeypatch):
    import json

    from src.checkpoint import save_checkpoint

    labels_path = tmp_path / "labels.json"
    monkeypatch.setattr("src.checkpoint.LABELS_PATH", labels_path)
    save_checkpoint(build_model(tiny=True), labels, path=tmp_path / "m.pt")

    assert json.loads(labels_path.read_text())["labels"] == labels


def test_checkpoint_metadata(tmp_path, labels):
    from src.checkpoint import checkpoint_metadata, save_checkpoint

    path = tmp_path / "m.pt"
    save_checkpoint(build_model(tiny=True), labels, path=path)
    metadata = checkpoint_metadata(path)
    assert metadata["labels"] == labels
    assert metadata["size_mb"] > 0


def test_checkpoint_metadata_missing_file(tmp_path):
    from src.checkpoint import checkpoint_metadata

    assert checkpoint_metadata(tmp_path / "absent.pt") == {}


def test_load_labels_falls_back_to_canonical(tmp_path):
    from src.config import EMOTION_LABELS
    from src.dataset import load_labels

    assert load_labels(tmp_path / "absent.json") == list(EMOTION_LABELS)


def test_load_labels_reads_file(tmp_path, labels):
    import json

    from src.dataset import load_labels

    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"labels": labels}))
    assert load_labels(path) == labels


def test_load_labels_tolerates_corrupt_json(tmp_path):
    from src.config import EMOTION_LABELS
    from src.dataset import load_labels

    path = tmp_path / "labels.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_labels(path) == list(EMOTION_LABELS)
