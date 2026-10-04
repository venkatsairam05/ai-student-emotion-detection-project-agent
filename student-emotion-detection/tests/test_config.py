"""Tests for configuration loading and the UI theme helpers."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def fresh_config(monkeypatch):
    """Reimport ``src.config`` so module-level env reads are re-evaluated."""

    def _reload(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, str(value))
        module = importlib.reload(importlib.import_module("src.config"))
        return module

    yield _reload
    for key in ("IMAGE_SIZE", "BATCH_SIZE", "EPOCHS", "SEED", "DEVICE"):
        monkeypatch.delenv(key, raising=False)
    importlib.reload(importlib.import_module("src.config"))


def test_env_helpers_parse_values(monkeypatch):
    from src.config import env_bool, env_float, env_int

    monkeypatch.setenv("TEST_INT", "7")
    monkeypatch.setenv("TEST_FLOAT", "1.5")
    monkeypatch.setenv("TEST_BOOL", "yes")
    assert env_int("TEST_INT", 0) == 7
    assert env_float("TEST_FLOAT", 0.0) == 1.5
    assert env_bool("TEST_BOOL", False) is True


def test_env_helpers_handle_missing(monkeypatch):
    from src.config import env_bool, env_float, env_int

    monkeypatch.delenv("TEST_MISSING", raising=False)
    assert env_int("TEST_MISSING", 3) == 3
    assert env_float("TEST_MISSING", 2.5) == 2.5
    assert env_bool("TEST_MISSING", True) is True


def test_env_helpers_handle_garbage(monkeypatch):
    from src.config import env_bool, env_float, env_int

    monkeypatch.setenv("TEST_INT", "abc")
    monkeypatch.setenv("TEST_FLOAT", "xyz")
    monkeypatch.setenv("TEST_BOOL", "")
    assert env_int("TEST_INT", 9) == 9
    assert env_float("TEST_FLOAT", 1.0) == 1.0
    assert env_bool("TEST_BOOL", False) is False


def test_env_bool_variants(monkeypatch):
    from src.config import env_bool

    for truthy in ("1", "true", "TRUE", "on", "Y"):
        monkeypatch.setenv("TEST_BOOL", truthy)
        assert env_bool("TEST_BOOL", False) is True
    for falsy in ("0", "false", "off", "no"):
        monkeypatch.setenv("TEST_BOOL", falsy)
        assert env_bool("TEST_BOOL", True) is False


def test_overrides_take_effect(fresh_config):
    module = fresh_config(IMAGE_SIZE=64, BATCH_SIZE=8, EPOCHS=2, SEED=7)
    assert module.IMAGE_SIZE == 64
    assert module.BATCH_SIZE == 8
    assert module.EPOCHS == 2
    assert module.SEED == 7


def test_ensure_dirs_is_idempotent():
    from src.config import ARTIFACTS_DIR, MODEL_DIR, ensure_dirs

    ensure_dirs()
    ensure_dirs()
    assert ARTIFACTS_DIR.is_dir()
    assert MODEL_DIR.is_dir()


def test_train_config_defaults_align_with_module():
    from src.config import BATCH_SIZE, EPOCHS, TrainConfig

    config = TrainConfig()
    assert config.epochs == EPOCHS
    assert config.batch_size == BATCH_SIZE


def test_train_config_to_dict():
    from src.config import TrainConfig

    payload = TrainConfig().to_dict()
    assert isinstance(payload, dict)
    assert payload["epochs"] > 0
    assert "augmentation" in payload


def test_model_config_to_dict():
    from src.config import ModelConfig

    payload = ModelConfig().to_dict()
    assert payload["num_classes"] == 7
    assert 0.0 <= payload["dropout"] <= 1.0


def test_labels_are_seven_unique():
    from src.config import EMOTION_LABELS

    assert len(EMOTION_LABELS) == 7
    assert len(set(EMOTION_LABELS)) == 7


def test_resolve_device_auto_returns_valid():
    from src.checkpoint import resolve_device

    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


def test_find_haarcascade_path_does_not_raise():
    from src.config import find_haarcascade_path

    path = find_haarcascade_path()
    assert path is not None


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #


def test_theme_covers_every_emotion(labels):
    from src.emotions import EMOTIONS
    from src.ui.theme import EMOTION_COLORS

    assert set(EMOTION_COLORS) == {e.name for e in EMOTIONS}


def test_theme_colors_are_hex():
    from src.ui.theme import BAND_COLORS, EMOTION_COLORS

    for color in list(EMOTION_COLORS.values()) + list(BAND_COLORS.values()):
        assert color.startswith("#")
        assert len(color) == 7


def test_theme_fallbacks():
    from src.ui.theme import FALLBACK_COLOR, band_color, color_scale, emotion_color

    assert emotion_color("unknown") == FALLBACK_COLOR
    assert band_color("unknown") == FALLBACK_COLOR
    assert color_scale("happy").startswith("rgba(")
    assert color_scale("happy", 0.5).endswith(", 0.5)")


def test_emotion_colors_are_distinct():
    """Every emotion needs its own colour, or the legend becomes unreadable."""

    from src.ui.theme import EMOTION_COLORS

    assert len(set(EMOTION_COLORS.values())) == len(EMOTION_COLORS)


def test_band_colors_are_distinct():
    from src.ui.theme import BAND_COLORS

    assert len(set(BAND_COLORS.values())) == len(BAND_COLORS)


def test_emotion_legend_contains_all_labels():
    from src.ui.theme import emotion_legend_markdown

    legend = emotion_legend_markdown().lower()
    for label in ("happy", "sad", "angry", "neutral", "surprise", "disgust", "fear"):
        assert label in legend


def test_charts_band_color_matches_theme():
    from src.ui.charts import band_color as charts_band
    from src.ui.theme import band_color as theme_band

    for band in ("High", "Moderate", "Low", "Very Low"):
        assert charts_band(band) == theme_band(band)


# --------------------------------------------------------------------------- #
# Animated HTML components
# --------------------------------------------------------------------------- #


def test_reduced_motion_is_honoured():
    """Animations must be disableable for users who ask for reduced motion."""

    from src.ui.theme import PAGE_CSS

    assert "prefers-reduced-motion" in PAGE_CSS
    assert "animation: none !important" in PAGE_CSS


def test_emotion_chips_pair_colour_with_text():
    """Colour is never the only cue, so chips must carry emoji and name."""

    from src.emotions import EMOTIONS
    from src.ui.theme import emotion_legend_markdown

    legend = emotion_legend_markdown()
    for emotion in EMOTIONS:
        assert emotion.emoji in legend
        assert emotion.name.title() in legend


def test_metric_cards_render_all_four_cards():
    from src.ui.theme import metric_cards_html

    html = metric_cards_html(
        {
            "mean_engagement": 0.72,
            "engagement_band": "High",
            "face_count": 5,
            "dominant_label": "happy",
        }
    )
    assert html.count('class="es-stat"') == 4
    assert "72%" in html
    assert "High" in html
    assert "5" in html


def test_band_gauge_percentage_is_clamped():
    from src.ui.theme import band_gauge_html

    assert "width:100%" in band_gauge_html(1.5)
    assert "width:0%" in band_gauge_html(-3.0)


def test_probability_bars_are_ordered_and_scaled():
    from src.config import EMOTION_LABELS
    from src.ui.theme import probability_bars_html

    probabilities = {label: 1.0 / len(EMOTION_LABELS) for label in EMOTION_LABELS}
    html = probability_bars_html(probabilities)
    assert html.count("es-bar-row") == len(EMOTION_LABELS)
    # Equal probabilities produce equal-width bars, all above zero.
    assert "width:14%" in html  # 1/7 rounded
    assert "width:0%" not in html


def test_probability_bars_handle_empty_input():
    from src.ui.theme import probability_bars_html

    assert probability_bars_html({}) == ""


def test_feature_cards_render_each_item():
    from src.ui.theme import feature_cards_html

    html = feature_cards_html([("A", "One", "first"), ("B", "Two", "second")])
    assert html.count('class="es-card"') == 2
    assert "One" in html and "Two" in html


def test_notice_tones_all_render():
    from src.ui.theme import notice_html

    for tone in ("info", "success", "warn", "error"):
        assert "es-notice" in notice_html("hello", tone=tone)


def test_mix_blends_toward_second_colour():
    from src.ui.theme import mix

    assert mix("#000000", "#ffffff", 0.0) == "#000000"
    assert mix("#000000", "#ffffff", 1.0) == "#ffffff"
    assert mix("#000000", "#ffffff", 0.5) == "#808080"


def test_css_has_no_unclosed_style_tag():
    from src.ui.theme import PAGE_CSS, SIDEBAR_CSS

    for css in (PAGE_CSS, SIDEBAR_CSS):
        assert css.count("<style>") == 1
        assert css.count("</style>") == 1
        assert css.count("{") == css.count("}")


def test_apply_theme_is_safe_without_streamlit():
    from src.ui.theme import apply_theme

    class FakeStreamlit:
        def markdown(self, *args, **kwargs):
            pass

        class _Sidebar:
            def markdown(self, *args, **kwargs):
                pass

        sidebar = _Sidebar()

    apply_theme(FakeStreamlit())
