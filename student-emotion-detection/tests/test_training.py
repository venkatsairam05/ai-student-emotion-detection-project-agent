"""Tests for the training loop and evaluation metrics."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from src.config import TrainConfig
from src.evaluation import (
    aggregate_metrics,
    confusion_dataframe_rows,
    confusion_matrix,
    evaluate_model,
    macro_f1,
    per_class_metrics,
    preprocess_pil,
    predict_labels,
    top_k_accuracy,
    weighted_f1,
)
from src.model import TinyEmotionCNN
from src.trainer import EarlyStopping, load_history, run_one_epoch, save_history, set_seed, train


@pytest.fixture
def tiny_loader(image_folder, image_size):
    from torch.utils.data import DataLoader

    from src.dataset import EmotionDataset, build_transforms

    train_tf, eval_tf = build_transforms(image_size)
    train_ds = EmotionDataset(image_folder / "train", transform=train_tf, labels=["happy", "sad"])
    val_ds = EmotionDataset(image_folder / "val", transform=eval_tf, labels=["happy", "sad"])
    return (
        DataLoader(train_ds, batch_size=2, shuffle=False),
        DataLoader(val_ds, batch_size=2, shuffle=False),
    )


# --------------------------------------------------------------------------- #
# Confusion matrix and metrics
# --------------------------------------------------------------------------- #


def test_confusion_matrix_counts():
    matrix = confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1], num_classes=2)
    assert matrix == [[1, 1], [0, 2]]


def test_confusion_matrix_rejects_length_mismatch():
    with pytest.raises(ValueError):
        confusion_matrix([0, 1], [0], num_classes=2)


def test_confusion_matrix_ignores_out_of_range():
    assert confusion_matrix([5], [0], num_classes=2) == [[0, 0], [0, 0]]


def test_per_class_metrics_perfect_prediction():
    matrix = [[3, 0], [0, 3]]
    metrics = per_class_metrics(matrix)
    assert metrics["0"]["precision"] == pytest.approx(1.0)
    assert metrics["1"]["f1"] == pytest.approx(1.0)
    assert metrics["0"]["support"] == 3


def test_per_class_metrics_zero_division_is_zero():
    metrics = per_class_metrics([[0, 0], [0, 0]])
    assert metrics["0"]["precision"] == 0.0
    assert metrics["0"]["recall"] == 0.0
    assert metrics["0"]["f1"] == 0.0


def test_macro_and_weighted_f1_agree_on_balanced():
    matrix = [[5, 0], [0, 5]]
    metrics = per_class_metrics(matrix)
    assert macro_f1(metrics) == pytest.approx(weighted_f1(metrics, matrix))


def test_macro_f1_weights_classes_equally():
    metrics = {"0": {"f1": 1.0, "support": 90.0}, "1": {"f1": 0.0, "support": 10.0}}
    assert macro_f1(metrics) == pytest.approx(0.5)


def test_aggregate_metrics_shape():
    result = aggregate_metrics([[2, 0], [1, 1]], top2=0.75)
    assert result["total"] == 4
    assert result["correct"] == 3
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["top2_accuracy"] == 0.75


def test_top_k_accuracy():
    logits = torch.tensor([[5.0, 1.0], [1.0, 5.0], [1.0, 5.0]])
    # argmax predictions are [0, 1, 1]; truth is [1, 0, 1] -> one hit.
    assert top_k_accuracy([1, 0, 1], logits, k=1) == pytest.approx(1 / 3)
    assert top_k_accuracy([1, 0, 1], logits, k=2) == pytest.approx(1.0)


def test_top_k_accuracy_empty():
    assert top_k_accuracy([], torch.zeros(0, 3), k=2) == 0.0


def test_confusion_dataframe_rows(labels):
    rows = confusion_dataframe_rows([[1, 0], [0, 1]], ["happy", "sad"])
    assert rows[0]["true"] == "happy"
    assert rows[0]["pred_happy"] == 1
    assert rows[0]["support"] == 1


# --------------------------------------------------------------------------- #
# Model evaluation
# --------------------------------------------------------------------------- #


def test_predict_labels_returns_one_row_per_image(tiny_model, batch_tensor, labels):
    results = predict_labels(tiny_model, batch_tensor, labels)
    assert len(results) == batch_tensor.shape[0]
    assert all(r["label"] in labels for r in results)
    assert all(0.0 <= r["engagement"] <= 1.0 for r in results)


def test_predict_labels_empty_batch(tiny_model, labels):
    assert predict_labels(tiny_model, torch.zeros(0, 3, 48, 48), labels) == []


def test_preprocess_pil_normalizes():
    from PIL import Image

    tensor = preprocess_pil(Image.new("RGB", (64, 64), (255, 255, 255)), 48)
    assert tensor.shape == (1, 3, 48, 48)
    assert float(tensor.max()) <= 1.5


def test_evaluate_model_over_loader(tiny_model, tiny_loader, labels):
    _, val_loader = tiny_loader
    metrics = evaluate_model(tiny_model, val_loader, labels, device="cpu")
    assert metrics["total"] == 2
    assert len(metrics["confusion_matrix"]) == len(labels)
    assert metrics["labels"] == labels


# --------------------------------------------------------------------------- #
# Early stopping
# --------------------------------------------------------------------------- #


def test_early_stopping_triggers_after_patience():
    stopper = EarlyStopping(patience=2)
    assert stopper.step(1.0, 1) is False
    assert stopper.step(0.9, 2) is False
    assert stopper.step(1.1, 3) is False
    assert stopper.step(1.2, 4) is True
    assert stopper.should_stop is True


def test_early_stopping_resets_counter_on_improvement():
    stopper = EarlyStopping(patience=3)
    stopper.step(1.0, 1)
    stopper.step(2.0, 2)
    stopper.step(0.5, 3)
    assert stopper.counter == 0
    assert stopper.best_epoch == 3


def test_early_stopping_respects_min_delta():
    # A 0.2 drop is below min_delta=0.5, so it does not count as improvement.
    stopper = EarlyStopping(patience=2, min_delta=0.5)
    stopper.step(1.0, 1)
    assert stopper.step(0.8, 2) is False
    assert stopper.best_epoch == 1


def test_early_stopping_zero_patience_never_stops():
    stopper = EarlyStopping(patience=0)
    stopper.step(1.0, 1)
    assert stopper.step(9.0, 2) is False


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def test_run_one_epoch_eval_does_not_change_weights(tiny_model, tiny_loader):
    import copy

    train_loader, val_loader = tiny_loader
    criterion = torch.nn.CrossEntropyLoss()
    before = copy.deepcopy(tiny_model.state_dict())
    run_one_epoch(tiny_model, val_loader, criterion, torch.device("cpu"))
    for key, value in tiny_model.state_dict().items():
        assert torch.allclose(value, before[key])


def test_run_one_epoch_training_updates_weights(tiny_loader):
    train_loader, _ = tiny_loader
    model = TinyEmotionCNN(num_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    before = [p.detach().clone() for p in model.parameters()]
    run_one_epoch(model, train_loader, criterion, torch.device("cpu"), optimizer=optimizer)
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters())
    )


def test_run_one_epoch_empty_loader():
    from torch.utils.data import DataLoader, TensorDataset

    empty = DataLoader(TensorDataset(torch.zeros(0, 3, 48, 48), torch.zeros(0, dtype=torch.long)))
    loss, accuracy = run_one_epoch(
        TinyEmotionCNN(num_classes=2), empty, torch.nn.CrossEntropyLoss(), torch.device("cpu")
    )
    assert (loss, accuracy) == (0.0, 0.0)


def test_train_runs_and_returns_history(tiny_loader):
    train_loader, val_loader = tiny_loader
    model = TinyEmotionCNN(num_classes=2)
    config = TrainConfig(epochs=2, patience=0, learning_rate=1e-3, use_class_weights=False)

    trained, history = train(model, train_loader, val_loader, config=config, device="cpu")

    assert len(history) == 2
    assert history[0].epoch == 1
    assert trained.training is False
    assert history[0].to_dict()["val_loss"] >= 0.0


def test_train_honors_early_stopping(tiny_loader):
    train_loader, val_loader = tiny_loader
    model = TinyEmotionCNN(num_classes=2)
    config = TrainConfig(epochs=12, patience=1, learning_rate=1e-3, use_class_weights=False)

    _, history = train(model, train_loader, val_loader, config=config, device="cpu")
    assert len(history) < 12


def test_train_progress_callback_fires(tiny_loader):
    train_loader, val_loader = tiny_loader
    events = []
    train(
        TinyEmotionCNN(num_classes=2),
        train_loader,
        val_loader,
        config=TrainConfig(epochs=1, patience=0, use_class_weights=False),
        device="cpu",
        progress_callback=lambda kind, payload: events.append(kind),
    )
    assert "epoch" in events


def test_save_and_load_history(tmp_path):
    from src.trainer import EpochMetrics

    metrics = [EpochMetrics(1, 0.5, 0.6, 0.4, 0.5, 1e-3)]
    path = save_history(metrics, tmp_path / "history.json")
    loaded = load_history(path)
    assert loaded[0]["epoch"] == 1
    assert loaded[0]["train_loss"] == pytest.approx(0.5)


def test_load_history_missing_file(tmp_path):
    assert load_history(tmp_path / "absent.json") == []


def test_load_history_corrupt_file(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("{ broken", encoding="utf-8")
    assert load_history(path) == []


def test_set_seed_is_reproducible():
    set_seed(123)
    first = torch.randn(5)
    set_seed(123)
    assert torch.allclose(first, torch.randn(5))


def test_history_json_is_valid(tmp_path):
    from src.trainer import EpochMetrics

    path = save_history([EpochMetrics(1, 0.1, 0.2, 0.3, 0.4, 5e-4)], tmp_path / "h.json")
    payload = json.loads(path.read_text())
    assert isinstance(payload, list)
    assert set(payload[0]) >= {"epoch", "train_loss", "val_accuracy"}
