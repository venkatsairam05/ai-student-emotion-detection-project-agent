"""Smoke tests for the CLI scripts.

Each script gets its argument parsing tested plus a real end-to-end run against a
tiny generated dataset, so a broken import or a bad CLI flag fails CI instead of
surfacing only when a user tries it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("PIL")

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


@pytest.fixture(scope="module")
def mini_dataset(tmp_path_factory):
    """A tiny synthetic image-folder dataset (3 images per class per split)."""

    from scripts.make_synthetic_data import generate

    out = tmp_path_factory.mktemp("synthetic")
    counts = generate(out_root=out, per_class=3, image_size=48, seed=1)
    assert counts["train"] == 21
    return out


def test_prepare_data_missing_csv_returns_one(tmp_path, capsys):
    import prepare_data

    assert prepare_data.main(["--csv", str(tmp_path / "absent.csv")]) == 1
    assert "not found" in capsys.readouterr().out


def test_prepare_data_converts_csv(tmp_path, capsys):
    import prepare_data

    csv_path = tmp_path / "fer2013.csv"
    pixels = " ".join(["10"] * (48 * 48))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        handle.write("emotion,pixels,Usage\n")
        handle.write(f"3,{pixels},Training\n")
        handle.write(f"0,{pixels},PublicTest\n")
        handle.write(f"6,{pixels},PrivateTest\n")

    out = tmp_path / "processed"
    assert prepare_data.main(["--csv", str(csv_path), "--out", str(out)]) == 0

    assert (out / "train" / "happy").is_dir()
    assert (out / "val" / "angry").is_dir()
    assert (out / "test" / "neutral").is_dir()
    assert len(list((out / "train" / "happy").glob("*.png"))) == 1
    assert "total images: 3" in capsys.readouterr().out


def test_prepare_data_counts_skipped_rows(tmp_path):
    import prepare_data

    csv_path = tmp_path / "fer2013.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        handle.write("emotion,pixels,Usage\n")
        handle.write("0,1 2 3,Training\n")
        handle.write("abc,4 5 6,Training\n")

    counters = prepare_data.convert(csv_path, out_root=tmp_path / "out")
    assert counters["_skipped"]["bad_pixel_count"] == 1
    assert counters["_skipped"]["bad_label"] == 1


def test_prepare_data_limit(tmp_path):
    import prepare_data

    csv_path = tmp_path / "fer2013.csv"
    pixels = " ".join(["10"] * (48 * 48))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        handle.write("emotion,pixels,Usage\n")
        for _ in range(5):
            handle.write(f"0,{pixels},Training\n")

    prepare_data.convert(csv_path, out_root=tmp_path / "out", limit=2)
    assert len(list((tmp_path / "out" / "train" / "angry").glob("*.png"))) == 2


def test_prepare_data_overwrite(tmp_path):
    import prepare_data

    csv_path = tmp_path / "fer2013.csv"
    pixels = " ".join(["10"] * (48 * 48))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        handle.write("emotion,pixels,Usage\n")
        handle.write(f"0,{pixels},Training\n")

    out = tmp_path / "out"
    prepare_data.convert(csv_path, out_root=out)
    stale = out / "train" / "happy" / "stale.png"
    stale.write_bytes(b"x")
    prepare_data.convert(csv_path, out_root=out, overwrite=True)
    assert not stale.exists()


def test_prepare_data_missing_file_raises(tmp_path):
    import prepare_data

    with pytest.raises(FileNotFoundError):
        prepare_data.convert(tmp_path / "absent.csv", out_root=tmp_path / "out")


def test_train_print_model(capsys):
    import train as train_script

    assert train_script.main(["--print-model", "--tiny"]) == 0
    output = capsys.readouterr().out
    assert "num_classes" in output
    assert "total trainable parameters" in output


def test_train_missing_data_returns_one(tmp_path, capsys):
    import train as train_script

    code = train_script.main(
        [
            "--train-dir", str(tmp_path / "nope"),
            "--val-dir", str(tmp_path / "nope"),
            "--no-save",
            "--tiny",
        ]
    )
    assert code == 1
    assert "prepare_data" in capsys.readouterr().out


def test_train_smoke_run(mini_dataset, tmp_path, capsys):
    import train as train_script

    checkpoint = tmp_path / "model.pt"
    history = tmp_path / "history.json"
    code = train_script.main(
        [
            "--train-dir", str(mini_dataset / "train"),
            "--val-dir", str(mini_dataset / "val"),
            "--epochs", "1",
            "--batch-size", "4",
            "--tiny",
            "--limit-batches", "2",
            "--checkpoint", str(checkpoint),
            "--history", str(history),
        ]
    )
    assert code == 0
    assert checkpoint.is_file()
    assert history.is_file()
    assert "validation results" in capsys.readouterr().out


def test_train_no_save(mini_dataset, tmp_path):
    import train as train_script

    checkpoint = tmp_path / "absent.pt"
    code = train_script.main(
        [
            "--train-dir", str(mini_dataset / "train"),
            "--val-dir", str(mini_dataset / "val"),
            "--epochs", "1",
            "--tiny",
            "--limit-batches", "1",
            "--no-save",
            "--checkpoint", str(checkpoint),
        ]
    )
    assert code == 0
    assert not checkpoint.exists()


@pytest.fixture(scope="module")
def trained_checkpoint(mini_dataset, tmp_path_factory):
    """Train a one-epoch tiny model once and share it across the module.

    Retraining per test dominated this module's runtime; one checkpoint is
    enough to cover every evaluate/predict assertion.
    """

    import train as train_script

    out = tmp_path_factory.mktemp("trained")
    checkpoint = out / "model.pt"
    code = train_script.main(
        [
            "--train-dir", str(mini_dataset / "train"),
            "--val-dir", str(mini_dataset / "val"),
            "--epochs", "1",
            "--batch-size", "4",
            "--tiny",
            "--checkpoint", str(checkpoint),
            "--history", str(out / "history.json"),
        ]
    )
    assert code == 0
    return checkpoint


def test_evaluate_missing_checkpoint_returns_one(tmp_path, capsys):
    import evaluate as evaluate_script

    assert evaluate_script.main(["--checkpoint", str(tmp_path / "absent.pt")]) == 1
    assert "No trained checkpoint" in capsys.readouterr().out


def test_evaluate_runs(mini_dataset, trained_checkpoint, tmp_path, capsys):
    import evaluate as evaluate_script

    capsys.readouterr()  # discard the training run's output
    confusion_csv = tmp_path / "confusion.csv"
    code = evaluate_script.main(
        [
            "--checkpoint", str(trained_checkpoint),
            "--test-dir", str(mini_dataset / "val"),
            "--batch-size", "4",
            "--save-confusion", str(confusion_csv),
        ]
    )
    assert code == 0
    assert confusion_csv.is_file()
    assert "macro F1" in capsys.readouterr().out


def test_evaluate_json_output(mini_dataset, trained_checkpoint, capsys):
    import evaluate as evaluate_script

    capsys.readouterr()
    assert evaluate_script.main(
        [
            "--checkpoint", str(trained_checkpoint),
            "--test-dir", str(mini_dataset / "val"),
            "--batch-size", "4",
            "--json",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "accuracy" in payload
    assert "confusion_matrix" in payload


def test_evaluate_missing_test_dir(mini_dataset, trained_checkpoint, tmp_path):
    import evaluate as evaluate_script

    assert evaluate_script.main(
        ["--checkpoint", str(trained_checkpoint), "--test-dir", str(tmp_path / "absent")]
    ) == 1


def test_predict_missing_source(tmp_path, capsys):
    import predict as predict_script

    assert predict_script.main(["--source", str(tmp_path / "absent")]) == 1
    assert "no images found" in capsys.readouterr().err


def test_predict_missing_checkpoint(mini_dataset, tmp_path, capsys):
    import predict as predict_script

    assert predict_script.main(
        ["--source", str(mini_dataset / "val"), "--checkpoint", str(tmp_path / "absent.pt")]
    ) == 1
    assert "No trained checkpoint" in capsys.readouterr().err


def test_predict_analyses_folder(mini_dataset, trained_checkpoint, tmp_path, capsys):
    import predict as predict_script

    capsys.readouterr()
    out_csv = tmp_path / "frames.csv"
    assert predict_script.main(
        [
            "--source", str(mini_dataset / "val"),
            "--checkpoint", str(trained_checkpoint),
            "--out", str(out_csv),
        ]
    ) == 0
    assert out_csv.is_file()
    captured = capsys.readouterr()
    assert "images analysed" in captured.err


def test_predict_json_output(mini_dataset, trained_checkpoint, tmp_path, capsys):
    import predict as predict_script

    capsys.readouterr()
    assert predict_script.main(
        [
            "--source", str(mini_dataset / "val"),
            "--checkpoint", str(trained_checkpoint),
            "--json",
            "--out", str(tmp_path / "f.csv"),
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["images"] > 0


def test_make_synthetic_data_cli(tmp_path, capsys):
    import make_synthetic_data

    out = tmp_path / "synth"
    assert make_synthetic_data.main(["--out", str(out), "--per-class", "2"]) == 0
    assert (out / "train").is_dir()
    assert (out / "val").is_dir()
    assert "synthetic" in capsys.readouterr().out


def test_synthetic_images_are_grayscale_sized(tmp_path):
    from scripts.make_synthetic_data import generate

    out = tmp_path / "synth"
    generate(out_root=out, per_class=1, image_size=48)
    from PIL import Image

    path = next((out / "train" / "happy").glob("*.png"))
    with Image.open(path) as image:
        assert image.size == (48, 48)
        assert image.mode == "L"


def test_cli_help_exits_cleanly():
    for module_name in ("prepare_data", "train", "evaluate", "predict", "make_synthetic_data"):
        module = __import__(module_name)
        with pytest.raises(SystemExit) as excinfo:
            module.parse_args(["--help"])
        assert excinfo.value.code == 0
