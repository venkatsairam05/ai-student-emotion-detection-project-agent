"""Dashboard rendering helpers: metrics, charts, and annotated images.

Every ``streamlit`` import lives inside a function so the module is importable
(and testable) without a Streamlit runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.analytics import SessionTracker, build_engagement_dataframe, build_student_table
from src.emotions import EMOTIONS, engagement_band
from src.evaluation import confusion_dataframe_rows
from src.ui import theme
from src.ui.theme import band_color, emotion_color


def st_or_none() -> Optional[Any]:
    """Return the ``streamlit`` module, or ``None`` when unavailable."""

    try:
        import streamlit as st  # type: ignore

        return st
    except Exception:
        return None


def render_html(st: Any, html: str) -> None:
    """Emit a pre-built HTML fragment, ignoring empty output."""

    if html:
        st.markdown(html, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def render_metric_cards(metrics: Dict[str, Any], columns: Optional[Sequence[Any]] = None) -> None:
    """Draw the headline metrics row: engagement, band, faces, dominant label.

    ``metrics`` is a :meth:`SessionTracker.snapshot` payload. No-ops when
    Streamlit is unavailable.
    """

    st = st_or_none()
    if st is None:
        return

    from src.ui.theme import metric_cards_html

    render_html(st, metric_cards_html(metrics))


def render_band_gauge(engagement: float, band: str = "") -> None:
    """Animated engagement gauge with band tick labels."""

    st = st_or_none()
    if st is None:
        return

    render_html(st, theme.band_gauge_html(engagement, band))


def render_feature_cards(items: Sequence[Tuple[str, str, str]]) -> None:
    """Hover-animated feature card grid."""

    st = st_or_none()
    if st is None:
        return

    render_html(st, theme.feature_cards_html(items))


def render_model_card(metadata: Dict[str, Any], labels: Sequence[str]) -> None:
    """Show checkpoint metadata so users know what model produced the numbers."""

    st = st_or_none()
    if st is None:
        return
    parameters = metadata.get("num_parameters")
    saved = metadata.get("saved_at")
    st.caption(
        f"Model: {parameters:,} parameters | trained {str(saved)[:10]} | "
        f"{len(labels)} classes: {', '.join(labels)}"
        if parameters
        else "No trained checkpoint loaded - predictions are random. Train one first."
    )


def render_trained_warning(is_trained: bool) -> None:
    """Warn when predictions come from untrained weights."""

    st = st_or_none()
    if st is None or is_trained:
        return
    render_html(
        st,
        theme.notice_html(
            "<strong>No trained checkpoint found</strong>, so emotion predictions are "
            "meaningless. Run <code>python scripts/prepare_data.py</code> then "
            "<code>python scripts/train.py</code> to get real results.",
            tone="warn",
        ),
    )


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #


def render_emotion_distribution(
    distribution: Dict[str, float],
    title: str = "Emotion distribution",
) -> None:
    """Animated horizontal bars for a label->share mapping."""

    st = st_or_none()
    if st is None or not distribution:
        return
    render_html(st, theme.section_header_html(title, "🍩"))
    render_html(st, theme.probability_bars_html(distribution))


def render_engagement_timeline(tracker: SessionTracker) -> None:
    """Line chart of engagement over the session, raw plus smoothed."""

    st = st_or_none()
    if st is None:
        return
    rows = build_engagement_dataframe(tracker)
    if not rows:
        render_html(st, theme.notice_html("No frames analysed yet.", tone="info"))
        return
    smoothed = tracker.smoothed_engagement()
    chart = {
        "Time (s)": [r["Time (s)"] for r in rows],
        "Engagement": [r["Engagement"] for r in rows],
    }
    if smoothed:
        chart["Smoothed"] = [round(v, 4) for _, v in smoothed]
    render_html(st, theme.section_header_html("Engagement over time", "📈"))
    if smoothed:
        latest = smoothed[-1][1]
        render_html(st, theme.band_gauge_html(latest))
    st.line_chart(chart, x="Time (s)", y=["Engagement", "Smoothed"] if smoothed else ["Engagement"])


def render_confusion_matrix(matrix: List[List[int]], labels: Sequence[str]) -> None:
    """Table view of a confusion matrix (rows = truth, columns = prediction)."""

    st = st_or_none()
    if st is None or not matrix:
        return
    st.dataframe(confusion_dataframe_rows(matrix, labels), use_container_width=True)


def render_student_table(tracker: SessionTracker) -> None:
    """Per-student engagement table."""

    st = st_or_none()
    if st is None:
        return
    rows = build_student_table(tracker)
    if not rows:
        render_html(
            st,
            theme.notice_html(
                "No student tracks yet &mdash; they appear once frames are analysed.",
                tone="info",
            ),
        )
        return
    render_html(st, theme.section_header_html("Per-student engagement", "🧑‍🎓"))
    st.dataframe(rows, use_container_width=True)


#: Alert severity -> icon and notice tone.
_ALERT_STYLES = {
    "high": ("🔴", "error"),
    "medium": ("🟠", "warn"),
    "low": ("🟡", "warn"),
}


def render_alerts(alerts: Sequence[Any]) -> None:
    """List classroom alerts with severity colouring."""

    st = st_or_none()
    if st is None:
        return
    if not alerts:
        render_html(
            st,
            theme.notice_html(
                "<strong>All clear.</strong> No classroom alerts raised in this session.",
                tone="success",
            ),
        )
        return
    for alert in alerts:
        data = alert.to_dict() if hasattr(alert, "to_dict") else dict(alert)
        severity = str(data.get("severity", ""))
        icon, tone = _ALERT_STYLES.get(severity, ("⚪", "info"))
        render_html(
            st,
            theme.notice_html(
                f"<strong>{data.get('kind', 'alert')}</strong> &mdash; "
                f"{data.get('message', '')}",
                tone=tone,
                icon=icon,
            ),
        )


def render_probability_panel(
    probabilities: Dict[str, float],
    title: str = "Class probabilities",
) -> None:
    """Animated per-label probability bars for a single frame or face."""

    st = st_or_none()
    if st is None or not probabilities:
        return
    render_html(st, theme.section_header_html(title, "🎯"))
    render_html(st, theme.probability_bars_html(probabilities))


def render_emotion_legend(labels: Optional[Sequence[str]] = None) -> None:
    """Hoverable legend mapping emotion names to their engagement weight."""

    st = st_or_none()
    if st is None:
        return
    wanted = None if labels is None else set(labels)
    chips = []
    for emotion in EMOTIONS:
        if wanted is not None and emotion.name not in wanted:
            continue
        distress = ' <span style="opacity:.8">&middot; distress</span>' if emotion.is_distress else ""
        chips.append(
            f'<span class="emotion-chip" style="background:{theme.emotion_gradient(emotion.name)};'
            f'color:{theme.emotion_color(emotion.name)}" '
            f'title="valence {emotion.valence:+.2f}, arousal {emotion.arousal:.2f}{distress}">'
            f"{emotion.emoji} {emotion.name.title()} "
            f'<span class="es-weight">{emotion.engagement_weight:.2f}</span></span>'
        )
    if not chips:
        return
    st.markdown(
        '<div style="margin-top:1.2rem">'
        '<div class="es-stat-label" style="margin-bottom:0.4rem">'
        "Engagement weight per emotion &middot; hover for detail</div>"
        f"{' '.join(chips)}</div>",
        unsafe_allow_html=True,
    )


def ordered_labels(mapping: Dict[str, Any]) -> List[str]:
    """Order keys by the canonical emotion list, extras appended alphabetically."""

    from src.config import EMOTION_LABELS

    canonical = [label for label in EMOTION_LABELS if label in mapping]
    extra = sorted(set(mapping).difference(canonical))
    return canonical + extra


#: Backwards-compatible private alias.
_ordered_labels = ordered_labels


def render_session_report(tracker: SessionTracker, report: Dict[str, Any]) -> None:
    """Full report view: summary numbers, timeline, heatmap, students, alerts."""

    st = st_or_none()
    if st is None:
        return

    render_html(st, theme.section_header_html("Session report", "📋"))
    render_html(
        st,
        theme.feature_cards_html(
            [
                ("🎞️", "Frames analysed", f"{int(report.get('frames_seen', 0)):,}"),
                ("🧑‍🎓", "Students tracked", f"{int(report.get('students_detected', 0)):,}"),
                (
                    "🎯",
                    "Mean engagement",
                    f"{float(report.get('mean_engagement', 0.0)):.0%}",
                ),
                ("🚨", "Alerts raised", f"{len(report.get('alerts', [])):,}"),
            ]
        ),
    )

    render_engagement_timeline(tracker)
    render_mood_heatmap(tracker)
    render_student_table(tracker)
    render_alerts(report.get("alerts", []))


def render_mood_heatmap(tracker: SessionTracker, labels: Optional[Sequence[str]] = None) -> None:
    """Stacked-bar view of how the emotion mix shifted over time."""

    from src.analytics import heatmap_frame, heatmap_matrix

    st = st_or_none()
    if st is None:
        return

    chosen = list(labels or tracker.labels or [e.name for e in EMOTIONS])
    matrix = heatmap_matrix(tracker, chosen, buckets=10)
    if not matrix:
        render_html(
            st,
            theme.notice_html(
                "Not enough frames for a mood timeline yet.", tone="info"
            ),
        )
        return
    render_html(st, theme.section_header_html("Mood mix over time", "🌈"))
    st.bar_chart(heatmap_frame(matrix, chosen), stack=True)


def render_export_buttons(tracker: SessionTracker) -> None:
    """CSV/JSON export buttons for the current session."""

    st = st_or_none()
    if st is None:
        return

    import json as _json

    from src.analytics import export_session_csv, export_session_json

    left, right = st.columns(2)
    with left:
        path = export_session_csv(tracker)
        st.download_button(
            "Download students CSV",
            data=Path(path).read_text(encoding="utf-8"),
            file_name="session_summary.csv",
            mime="text/csv",
        )
    with right:
        path = export_session_json(tracker)
        st.download_button(
            "Download full report JSON",
            data=_json.dumps(tracker.report(), indent=2, default=str),
            file_name="session_report.json",
            mime="application/json",
        )


def render_annotated_image(image: Any, caption: Optional[str] = None) -> None:
    """Display an annotated BGR frame as RGB inside a glowing frame."""

    st = st_or_none()
    if st is None or image is None:
        return
    try:
        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception:
        rgb = image
    st.image(rgb, caption=caption, use_container_width=True)


def image_grid(items: Sequence[Tuple[Any, str]], columns: int = 3) -> None:
    """Lay out annotated images in a responsive, hover-animated grid."""

    st = st_or_none()
    if st is None or not items:
        return
    palette = ["#7c3aed", "#ec4899", "#f59e0b", "#22d3ee", "#34e5a2", "#4f7dff"]
    rows = [items[i : i + columns] for i in range(0, len(items), columns)]
    for row_index, row in enumerate(rows):
        for slot, column in enumerate(st.columns(columns)):
            with column:
                if slot >= len(row):
                    continue
                image, caption = row[slot]
                accent = palette[row_index * columns + slot % len(palette)]
                st.markdown(
                    '<div class="es-card" style="padding:0.55rem;'
                    f'--es-accent:{accent};--es-glow:{theme.color_scale(accent, 0.3)};'
                    f'animation-delay:{slot * 70}ms">',
                    unsafe_allow_html=True,
                )
                render_annotated_image(image, caption)
                st.markdown("</div>", unsafe_allow_html=True)
