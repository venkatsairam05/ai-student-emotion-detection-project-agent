"""Student Emotion Detection for Smart Classrooms.

A from-scratch CNN reads student facial expressions and turns them into
classroom-level engagement signals.

Run with::

    streamlit run app.py

The module is deliberately thin: page routing lives in ``src.ui.pages``,
charting in ``src.ui.charts``, and the model is loaded once per process and
cached.
"""

from __future__ import annotations

import streamlit as st

from src.config import (
    CHECKPOINT_PATH,
    DETECTION_CONFIDENCE,
    DETECTION_MIN_SIZE,
    MAX_FACES,
    MIN_CONFIDENCE,
    TRAIN_HISTORY_PATH,
)
from src.ui import pages, theme


@st.cache_resource(show_spinner="Loading emotion model...")
def load_predictor(checkpoint: str, device: str, max_faces: int, min_confidence: float):
    """Load the trained checkpoint once and reuse it across reruns.

    ``st.cache_resource`` (not ``cache_data``) is required here because the
    model object is not serializable and must stay the same instance.
    """

    from src.detector import EmotionPredictor, FaceDetector

    detector = FaceDetector(
        min_confidence=DETECTION_CONFIDENCE,
        min_size=DETECTION_MIN_SIZE,
        max_faces=max_faces,
    )
    try:
        predictor = EmotionPredictor.from_checkpoint(
            checkpoint, device=device, detector=detector, min_confidence=min_confidence
        )
        return predictor, True
    except FileNotFoundError:
        return _untrained_predictor(detector, min_confidence), False


def _untrained_predictor(detector, min_confidence: float):
    """Build a predictor on random weights so the UI still renders.

    Used only when no checkpoint exists; the UI shows a prominent warning and
    the numbers are meaningless by construction.
    """

    from src.dataset import load_labels
    from src.detector import EmotionPredictor
    from src.model import build_model

    labels = load_labels()
    model = build_model()
    model.eval()
    predictor = EmotionPredictor(
        model=model,
        labels=labels,
        detector=detector,
        min_confidence=min_confidence,
        smooth=True,
    )
    predictor._trained_flag = False
    predictor.metadata = {"untrained": True}
    return predictor


@st.cache_resource(show_spinner=False)
def load_model_page_data() -> dict:
    """Checkpoint metadata plus training history for the Model page."""

    from src.checkpoint import checkpoint_metadata

    from src.trainer import load_history

    return {
        "checkpoint": checkpoint_metadata(CHECKPOINT_PATH),
        "history": load_history(TRAIN_HISTORY_PATH),
    }


def main() -> None:
    """Entry point: configure the page, load the model, route to a page."""

    st.set_page_config(
        page_title="EduSense · Student Emotion Detection",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    theme.apply_theme(st)

    st.markdown(
        theme.notice_html(
            "Colour and motion here are decoration. Every value is also written as text, "
            "and all animation stops if your system asks for reduced motion.",
            tone="info",
            icon="🎨",
        ),
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("🎓 EduSense")
        st.caption("Classroom engagement from facial expressions.")
        st.markdown(
            '<div class="es-pill good"><span class="es-dot"></span>Live &middot; local only</div>',
            unsafe_allow_html=True,
        )
        st.divider()

    predictor, trained = load_predictor(
        str(CHECKPOINT_PATH), "auto", MAX_FACES, MIN_CONFIDENCE
    )
    metadata = getattr(predictor, "metadata", {}) or {}

    pages.render_header(st, predictor, metadata if trained else {})

    settings = pages.render_sidebar(st)

    page_key = st.sidebar.radio(
        "Navigate",
        list(pages.PAGES.keys()),
        format_func=lambda key: pages.PAGES[key],
    )

    detector = predictor.detector
    detector.max_faces = settings["max_faces"]

    if page_key == "live":
        pages.render_live_page(st, predictor, settings, detector)
    elif page_key == "images":
        pages.render_images_page(st, predictor, settings)
    elif page_key == "video":
        pages.render_video_page(st, predictor, settings)
    elif page_key == "model":
        pages.render_model_page(st)
    else:
        pages.render_method_page(st)

    st.divider()
    from src.ui import charts

    charts.render_emotion_legend()


if __name__ == "__main__":
    main()
