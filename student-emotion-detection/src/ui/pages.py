"""Dashboard pages: live camera, image analysis, and model evaluation.

Each page is a plain function that takes ``streamlit`` as a parameter. That
keeps page logic importable from tests and lets ``app.py`` stay a thin entry
point.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.analytics import AlertConfig, SessionTracker, export_frame_csv
from src.config import DETECTION_CONFIDENCE, MAX_FACES, MIN_CONFIDENCE, SMOOTHING_FACTOR
from src.ui import charts, theme

#: Page key -> (label, emoji) for the sidebar radio.
PAGES: Dict[str, str] = {
    "live": "📡 Live classroom",
    "images": "🖼️ Image analysis",
    "video": "🎬 Video timeline",
    "model": "🧠 Model & training",
    "method": "🧭 Method",
}


# --------------------------------------------------------------------------- #
# Sidebar / header
# --------------------------------------------------------------------------- #


def render_header(st: Any, predictor: Optional[Any], metadata: Dict[str, Any]) -> None:
    """Animated hero banner, trained-model notice, and model metadata."""

    st.markdown(
        theme.hero_html(
            "Student Emotion Detection",
            "A from-scratch CNN reads student facial expressions and turns them "
            "into classroom-level engagement signals.",
            status="Model ready" if metadata else "Untrained model",
            status_tone="good" if metadata else "warn",
        ),
        unsafe_allow_html=True,
    )
    charts.render_trained_warning(bool(metadata))
    charts.render_model_card(metadata, getattr(predictor, "labels", []) or [])


def section(st: Any, title: str, icon: str = "") -> None:
    """Render an animated section header."""

    st.markdown(theme.section_header_html(title, icon), unsafe_allow_html=True)


def render_sidebar(st: Any) -> Dict[str, Any]:
    """Controls for detection, smoothing, and thresholds."""

    with st.sidebar:
        st.header("Settings")
        max_faces = st.slider("Max faces per frame", 1, 60, MAX_FACES, step=1)
        min_confidence = st.slider(
            "Minimum confidence",
            0.0,
            1.0,
            MIN_CONFIDENCE,
            0.05,
            help="Below this, the face is flagged as uncertain rather than hidden.",
        )
        smoothing = st.slider(
            "Temporal smoothing",
            0.0,
            1.0,
            SMOOTHING_FACTOR,
            0.05,
            help="Exponential moving average over frames. Higher is steadier but slower.",
        )
        show_labels = st.checkbox("Draw labels on video", value=True)
        st.divider()
        st.header("Alerts")
        distress_share = st.slider(
            "Distress alert threshold", 0.1, 1.0, 0.4, 0.05,
            help="Fraction of visible faces showing a distress emotion.",
        )
        cooldown = st.slider("Alert cooldown (s)", 1.0, 60.0, 8.0, 1.0)
        st.caption(
            "Faces are analysed locally. Nothing is uploaded, and no identity "
            "is inferred - only the emotion class and its confidence."
        )
    return {
        "max_faces": int(max_faces),
        "min_confidence": float(min_confidence),
        "smoothing": float(smoothing),
        "show_labels": bool(show_labels),
        "alert_config": AlertConfig(
            distress_share=float(distress_share), cooldown_seconds=float(cooldown)
        ),
    }


# --------------------------------------------------------------------------- #
# Live classroom
# --------------------------------------------------------------------------- #


def get_tracker(st: Any, alert_config: AlertConfig, labels: Optional[List[str]] = None) -> SessionTracker:
    """Fetch (or create) the session tracker stored in Streamlit's session state."""

    tracker = st.session_state.get("tracker")
    if tracker is None:
        tracker = SessionTracker(labels=labels or [], alert_config=alert_config)
        st.session_state["tracker"] = tracker
    else:
        tracker.alert_config = alert_config
    return tracker


def render_live_page(
    st: Any,
    predictor: Any,
    settings: Dict[str, Any],
    detector: Any,
) -> None:
    """Live webcam page using the local camera bridge."""

    from src.camera import get_camera_bridge

    section(st, "Live classroom", "📡")
    st.markdown(
        theme.notice_html(
            "The browser captures frames with <code>getUserMedia</code> and posts them to a "
            "local bridge process. <strong>Frames never leave your machine.</strong>",
            tone="info",
            icon="🔒",
        ),
        unsafe_allow_html=True,
    )

    if predictor is None:
        st.error("No model available. Train one, or run the synthetic-data smoke test.")
        return

    left, right = st.columns([3, 2])

    with left:
        bridge = get_camera_bridge()
        st.caption(f"Camera bridge: `{bridge.url}`")
        st.markdown('<div class="es-card" style="padding:0.5rem">', unsafe_allow_html=True)
        st.components.v1.iframe(bridge.url, height=420)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        section(st, "Live readout", "🧠")
        tracker = get_tracker(st, settings["alert_config"], list(getattr(predictor, "labels", [])))
        start = st.button("Analyse latest frame", type="primary", key="analyse_frame")

        latest = bridge.buffer.latest()
        if latest is None:
            st.markdown(
                theme.notice_html(
                    "Press <strong>Start camera</strong> in the panel to begin streaming frames.",
                    tone="warn",
                    icon="📷",
                ),
                unsafe_allow_html=True,
            )
        else:
            seq, jpeg, timestamp = latest
            st.caption(f"frame seq {seq} - buffered {len(bridge.buffer)}")

            result = predictor.predict(jpeg, apply_smoothing=settings["smoothing"] > 0)
            annotated = predictor.annotate(jpeg, result, show_labels=settings["show_labels"])
            charts.render_annotated_image(
                annotated,
                caption=(
                    f"{result.face_count} face(s) | engagement {result.mean_engagement:.0%} "
                    f"({result.engagement_band}) | {result.dominant_label}"
                ),
            )

            snapshot = tracker.update(result, timestamp=time.time())
            charts.render_metric_cards(snapshot.to_dict())
            charts.render_band_gauge(
                snapshot.mean_engagement, snapshot.engagement_band
            )

            section(st, "Per-face predictions", "👤")
            if result.faces:
                st.dataframe(
                    [
                        {
                            "Box (x, y, w, h)": str(face.region.as_tuple()),
                            "Emotion": f"{face.emoji} {face.label}",
                            "Confidence": round(face.confidence, 3),
                            "Engagement": round(face.engagement, 3),
                            "Band": face.band,
                            "Certain": "yes" if face.certain else "low",
                        }
                        for face in result.faces
                    ],
                    use_container_width=True,
                )
                charts.render_probability_panel(
                    result.faces[0].probabilities, title="First face - class probabilities"
                )
            else:
                st.markdown(
                    theme.notice_html(
                        "No faces detected in this frame. Try moving into frame or "
                        "improving the lighting.",
                        tone="warn",
                        icon="🕶️",
                    ),
                    unsafe_allow_html=True,
                )

            if start:
                st.session_state["last_snapshot"] = snapshot.to_dict()

    st.divider()
    section(st, "Session so far", "🗂️")
    charts.render_engagement_timeline(st.session_state.get("tracker"))
    charts.render_mood_heatmap(st.session_state.get("tracker"))
    charts.render_student_table(st.session_state.get("tracker"))
    charts.render_alerts(st.session_state.get("tracker").alerts if st.session_state.get("tracker") else [])
    charts.render_export_buttons(st.session_state.get("tracker"))

    if st.button("Reset session"):
        if st.session_state.get("tracker"):
            st.session_state["tracker"].reset()
        st.success("Session cleared.")


def render_video_page(st: Any, predictor: Any, settings: Dict[str, Any]) -> None:
    """Analyse an uploaded classroom video and plot the engagement timeline."""

    from src.camera import analyse_video

    section(st, "Video timeline", "🎬")
    if predictor is None:
        st.error("No model available. Train one first.")
        return

    upload = st.file_uploader("Upload a lecture video", type=["mp4", "mov", "avi", "mkv", "webm"])
    stride = st.slider("Sample every N frames", 1, 60, 10, 1,
                       help="Larger strides analyse faster but skip detail.")

    if upload is None:
        st.markdown(
            theme.notice_html(
                "Upload a short clip &mdash; a 1-3 minute sample is plenty &mdash; to see "
                "the engagement curve.",
                tone="info",
                icon="🎞️",
            ),
            unsafe_allow_html=True,
        )
        return

    data = upload.getvalue()
    st.video(data)

    if not st.button("Analyse video", type="primary"):
        return

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
        handle.write(data)
        temp_path = handle.name

    progress = st.progress(0.0, text="Analysing frames...")
    try:
        analysis = analyse_video(
            temp_path,
            predictor,
            stride=stride,
            progress_callback=lambda index, total: progress.progress(
                min(0.99, index / total) if total else 0.0
            ),
        )
    except (ImportError, FileNotFoundError) as exc:
        st.error(str(exc))
        return
    finally:
        Path(temp_path).unlink(missing_ok=True)

    progress.progress(1.0, text="Done")
    summary = analysis.summary()
    from src.emotions import engagement_band as _band_of

    mean_engagement = float(summary["mean_engagement"])
    charts.render_metric_cards(
        {
            "mean_engagement": mean_engagement,
            "engagement_band": _band_of(mean_engagement),
            "face_count": int(round(summary["avg_faces"])),
            "dominant_label": summary["dominant_label"],
        }
    )
    charts.render_band_gauge(mean_engagement, _band_of(mean_engagement))

    section(st, "Engagement over the clip", "📈")
    curve = analysis.engagement_curve()
    st.line_chart(
        {
            "Time (s)": [round(t, 2) for t, _ in curve],
            "Engagement": [round(v, 4) for _, v in curve],
        },
        x="Time (s)",
        y="Engagement",
    )

    section(st, "Mood mix over time", "🌈")
    frames = [
        {"bucket": i, **{k: v for k, v in sample.histogram.items()}}
        for i, sample in enumerate(analysis.samples)
    ]
    if frames:
        columns = [c for c in frames[0] if c != "bucket"]
        st.bar_chart(
            [{k: float(row.get(k, 0)) for k in columns} for row in frames],
            x="bucket",
            stack=True,
        )
        # Animated version of the same numbers, coloured per emotion.
        charts.render_emotion_distribution(
            {
                column: sum(float(row.get(column, 0)) for row in frames)
                for column in columns
            },
            title="Total mood share across the clip",
        )

    with st.expander("Raw summary JSON"):
        st.json(summary)


def render_images_page(st: Any, predictor: Any, settings: Dict[str, Any]) -> None:
    """Batch analysis of uploaded photos or a folder of images."""

    from src.camera import list_images

    section(st, "Image analysis", "🖼️")
    if predictor is None:
        st.error("No model available. Train one first.")
        return

    uploads = st.file_uploader(
        "Upload classroom photos",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        accept_multiple_files=True,
    )
    folder = st.text_input("...or a local folder path", value="")

    sources: List[Any] = []
    if uploads:
        sources = [(f.name, f.getvalue()) for f in uploads]
    elif folder.strip():
        paths = list_images(Path(folder.strip()))
        if not paths:
            st.warning(f"No images found in {folder.strip()}")
        sources = [(p.name, p.read_bytes()) for p in paths[:100]]

    if not sources:
        st.markdown(
            theme.notice_html(
                "Upload one or more photos to analyse.", tone="info", icon="📤"
            ),
            unsafe_allow_html=True,
        )
        return

    if not st.button(f"Analyse {len(sources)} image(s)", type="primary"):
        return

    tracker = get_tracker(st, settings["alert_config"], list(getattr(predictor, "labels", [])))
    rows: List[Dict[str, Any]] = []
    previews: List[Tuple[Any, str]] = []
    annotated_dir = Path("artifacts") / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    for index, (name, raw) in enumerate(sources, start=1):
        result = predictor.predict(raw, apply_smoothing=False)
        payload = result.to_dict()
        payload["source"] = name
        rows.append(payload)
        tracker.update(result, timestamp=float(index))
        previews.append(
            (
                predictor.annotate(raw, result, show_labels=settings["show_labels"]),
                f"{name} - {result.mean_engagement:.0%} {result.dominant_label}",
            )
        )
        from src.detector import save_frame

        save_frame(previews[-1][0], annotated_dir / f"{Path(name).stem}_annotated.png")
        if index % 10 == 0:
            st.progress(min(1.0, index / len(sources)), text=f"Analysed {index}/{len(sources)}")

    st.markdown(
        theme.notice_html(
            f"<strong>Analysed {len(rows)} image(s).</strong>", tone="success"
        ),
        unsafe_allow_html=True,
    )

    summary = {
        "images": len(rows),
        "faces_detected": sum(r["face_count"] for r in rows),
        "mean_engagement": sum(r["mean_engagement"] for r in rows) / len(rows),
    }
    mean_engagement = float(summary["mean_engagement"])
    charts.render_metric_cards(
        {
            "mean_engagement": mean_engagement,
            "engagement_band": _band(mean_engagement),
            "face_count": summary["faces_detected"],
            "dominant_label": _dominant(rows),
        }
    )
    charts.render_band_gauge(mean_engagement, _band(mean_engagement))

    charts.render_emotion_distribution(
        _dominant_counts(rows),
        title="Which emotions dominated across the batch",
    )

    section(st, "Annotated images", "🖌️")
    charts.image_grid(previews[:9], columns=3)

    section(st, "Per-image results", "📋")
    st.dataframe(
        [
            {
                "Image": r["source"],
                "Faces": r["face_count"],
                "Engagement": round(r["mean_engagement"], 3),
                "Band": r["engagement_band"],
                "Dominant": r["dominant_label"],
                "Distressed": r["distress_count"],
            }
            for r in rows
        ],
        use_container_width=True,
    )

    csv_path = export_frame_csv(rows, Path("reports") / "frame_analysis.csv")
    st.download_button(
        "Download results CSV",
        data=Path(csv_path).read_text(encoding="utf-8"),
        file_name="frame_analysis.csv",
        mime="text/csv",
    )

    charts.render_export_buttons(tracker)


def _band(engagement: float) -> str:
    from src.emotions import engagement_band

    return engagement_band(engagement)


def _dominant_counts(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Share of images whose dominant emotion was each label.

    Shares are normalised so the animated bars read as percentages of the batch.
    """

    counts: Dict[str, float] = {}
    for row in rows:
        label = str(row.get("dominant_label", "neutral"))
        counts[label] = counts.get(label, 0.0) + 1.0
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {label: value / total for label, value in counts.items()}


def _dominant(rows: List[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for row in rows:
        label = str(row.get("dominant_label", "neutral"))
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "neutral"
    return max(sorted(counts.items()), key=lambda kv: kv[1])[0]


# --------------------------------------------------------------------------- #
# Model & training
# --------------------------------------------------------------------------- #


def render_model_page(st: Any) -> None:
    """Model architecture, checkpoint metadata, and training history."""

    from src.checkpoint import CHECKPOINT_PATH, checkpoint_metadata
    from src.config import TRAIN_HISTORY_PATH
    from src.model import build_model, describe_architecture, layer_summary
    from src.trainer import load_history

    section(st, "Model & training", "🧠")

    model = build_model()
    architecture = describe_architecture(model)
    total_parameters = int(architecture.get("total_parameters", 0))

    charts.render_feature_cards(
        [
            ("🏗️", "Architecture", "VGG-style CNN with BatchNorm, max pooling, and a global-average-pooling head."),
            ("🔢", "Parameters", f"{total_parameters:,} weights, trained from scratch on FER2013."),
            ("📐", "Input", f"48x48 RGB crops, matching FER2013's native resolution."),
            ("🏷️", "Classes", "Seven expressions: angry, disgust, fear, happy, sad, surprise, neutral."),
        ]
    )

    section(st, "Architecture detail", "🔍")
    with st.expander("Architecture summary and layer listing"):
        st.json(architecture)
        st.code("\n".join(layer_summary(model)), language="text")

    section(st, "Checkpoint", "💾")
    metadata = checkpoint_metadata(CHECKPOINT_PATH)
    if metadata:
        with st.expander("Checkpoint metadata"):
            st.json(metadata)
    else:
        st.markdown(
            theme.notice_html(
                f"No checkpoint at <code>{CHECKPOINT_PATH}</code>. Train one with "
                "<code>python scripts/prepare_data.py</code> then "
                "<code>python scripts/train.py</code>.",
                tone="warn",
            ),
            unsafe_allow_html=True,
        )

    section(st, "Training history", "📜")
    history = load_history(TRAIN_HISTORY_PATH)
    if not history:
        st.markdown(
            theme.notice_html(
                f"No history file at <code>{TRAIN_HISTORY_PATH}</code> yet.",
                tone="info",
                icon="📭",
            ),
            unsafe_allow_html=True,
        )
        return

    st.markdown("##### Loss")
    st.line_chart(
        {
            "epoch": [h["epoch"] for h in history],
            "train_loss": [h["train_loss"] for h in history],
            "val_loss": [h["val_loss"] for h in history],
        },
        x="epoch",
    )

    st.markdown("##### Accuracy")
    st.line_chart(
        {
            "epoch": [h["epoch"] for h in history],
            "train_accuracy": [h["train_accuracy"] for h in history],
            "val_accuracy": [h["val_accuracy"] for h in history],
        },
        x="epoch",
    )

    best = max(history, key=lambda h: h.get("val_accuracy", 0.0))
    accuracy = float(best.get("val_accuracy", 0.0))
    charts.render_feature_cards(
        [
            ("🏅", "Best validation accuracy", f"{accuracy:.2%}"),
            ("📅", "Best epoch", str(best.get("epoch", "-"))),
            ("🔁", "Epochs run", str(len(history))),
            ("📉", "Best validation loss", f"{float(best.get('val_loss', 0.0)):.4f}"),
        ]
    )


def render_method_page(st: Any) -> None:
    """Explain the pipeline, the engagement mapping, and the ethical caveats."""

    from src.emotions import DISTRESS_LABELS, EMOTIONS

    section(st, "How this works", "🧭")
    charts.render_feature_cards(
        [
            ("👤", "1. Face detection", "An OpenCV Haar cascade finds frontal faces. Without it the whole frame is treated as one region."),
            ("✂️", "2. Crop and resize", "Each face is padded, converted to RGB, and resized to 48x48 to match FER2013."),
            ("🧠", "3. Classification", "A from-scratch VGG-style CNN outputs one of seven emotion logits."),
            ("〰️", "4. Smoothing", "Per-face probabilities pass through an IoU-tracked moving average, killing frame flicker."),
            ("🎯", "5. Engagement", "Each emotion carries a declared weight; the weighted mean becomes the engagement score."),
            ("🗂️", "6. Aggregation", "Frames roll into student tracks and a timeline, with cooldown-gated distress alerts."),
        ]
    )

    st.markdown(
        theme.notice_html(
            "The mapping from expression to engagement is a judgement call, not a fact, "
            "so it is declared explicitly in <code>src/emotions.py</code> rather than hidden "
            "inside a learned layer. That way it can be argued with and changed without "
            "retraining.",
            tone="info",
            icon="🧭",
        ),
        unsafe_allow_html=True,
    )

    section(st, "Emotion weights", "⚖️")
    st.dataframe(
        [
            {
                "Emotion": f"{e.emoji} {e.name}",
                "Valence": e.valence,
                "Arousal": e.arousal,
                "Engagement weight": e.engagement_weight,
                "Distress signal": "yes" if e.is_distress else "no",
            }
            for e in EMOTIONS
        ],
        use_container_width=True,
    )

    section(st, "Limitations you should know", "⚠️")
    st.markdown(
        theme.notice_html(
            "<strong>A facial expression is a weak proxy for engagement.</strong> A student "
            "may look neutral because they are concentrating, or angry because a joke landed. "
            "Treat the output as one signal among many, never as a diagnosis or a graded "
            "assessment.",
            tone="warn",
            icon="⚠️",
        ),
        unsafe_allow_html=True,
    )
    charts.render_feature_cards(
        [
            ("🎭", "Dataset bias", "FER2013 is posed and noisy. Real classrooms differ in camera, angle, lighting, and demographics."),
            ("🧩", "Expression is not understanding", "A frown does not tell you whether the student followed the explanation."),
            ("🕶️", "No identity tracking", "Faces are tracked within a session only to avoid double-counting. Nothing identifies a student."),
            ("✍️", "Consent matters", "Pointing a camera at students requires informed consent from them and their guardians."),
            ("🩺", "Not diagnostic", f"Distress labels ({', '.join(DISTRESS_LABELS)}) are cues for a human to check in, not conclusions."),
        ]
    )

    section(st, "What would make this better", "🚀")
    charts.render_feature_cards(
        [
            ("👁️", "Add attention signals", "Gaze direction, blink rate, and posture so engagement is not inferred from expression alone."),
            ("🎓", "Validate on real classrooms", "Collect consented classroom ground truth instead of reporting only FER2013 numbers."),
            ("🎚️", "Calibrate per camera", "Tune thresholds per camera and per lighting condition."),
            ("⏱️", "Aggregate longer", "A two-second engagement score is inherently noisy; longer windows are steadier."),
        ]
    )
