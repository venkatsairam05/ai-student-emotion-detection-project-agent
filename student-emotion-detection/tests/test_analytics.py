"""Tests for classroom analytics: session tracking, alerts, and reports."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")

from src.analytics import (
    AlertConfig,
    SessionTracker,
    build_engagement_dataframe,
    build_student_table,
    export_frame_csv,
    export_session_csv,
    export_session_json,
    heatmap_frame,
    heatmap_matrix,
)


@pytest.fixture
def tracker(labels):
    return SessionTracker(labels=labels)


def test_empty_session_is_neutral(tracker):
    report = tracker.report()
    assert report["frames_seen"] == 0
    assert report["students_detected"] == 0
    assert report["mean_engagement"] == 0.0
    assert report["alerts"] == []


def test_update_returns_snapshot(tracker, make_frame_result, make_face_prediction):
    result = make_frame_result([make_face_prediction("happy", 0.9), make_face_prediction("happy", 0.85)])
    snapshot = tracker.update(result, timestamp=1.0)
    assert snapshot.face_count == 2
    assert snapshot.mean_engagement > 0.5
    assert snapshot.frames_seen == 1


def test_session_tracks_students(tracker, make_frame_result, make_face_prediction):
    for index in range(3):
        result = make_frame_result(
            [
                make_face_prediction("happy", 0.9, box=(10, 10, 40, 40)),
                make_face_prediction("sad", 0.3, box=(200, 10, 40, 40)),
            ]
        )
        tracker.update(result, timestamp=float(index))

    assert len(tracker.tracks) == 2
    report = tracker.report()
    assert report["students_detected"] == 2
    assert report["duration_seconds"] == pytest.approx(2.0)


def test_tracks_persist_for_stationary_face(tracker, make_frame_result, make_face_prediction):
    for index in range(4):
        result = make_frame_result([make_face_prediction("happy", 0.9, box=(50, 50, 40, 40))])
        tracker.update(result, timestamp=float(index))
    assert len(tracker.tracks) == 1


def test_student_dominant_label_and_engagement(tracker, make_frame_result, make_face_prediction):
    for index in range(3):
        tracker.update(
            make_frame_result([make_face_prediction("happy", 0.9, box=(0, 0, 40, 40))]),
            timestamp=float(index),
        )
    track = next(iter(tracker.tracks.values()))
    assert track.dominant_label() == "happy"
    assert track.mean_engagement() == pytest.approx(0.9)
    assert track.mean_confidence() == pytest.approx(0.9)
    assert track.distress_share() == 0.0


def test_student_distress_share(tracker, make_frame_result, make_face_prediction):
    tracker.update(
        make_frame_result([make_face_prediction("angry", 0.1, box=(0, 0, 40, 40))]),
        timestamp=0.0,
    )
    track = next(iter(tracker.tracks.values()))
    assert track.distress_share() == pytest.approx(1.0)


def test_label_distribution_sums_to_one(tracker, make_frame_result, make_face_prediction):
    tracker.update(
        make_frame_result(
            [
                make_face_prediction("happy", 0.9, box=(0, 0, 40, 40)),
                make_face_prediction("sad", 0.3, box=(100, 0, 40, 40)),
            ]
        ),
        timestamp=0.0,
    )
    distribution = tracker.label_distribution()
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_trend_detects_drop(tracker, make_frame_result, make_face_prediction):
    for index in range(10):
        tracker.update(make_frame_result([make_face_prediction("happy", 0.95)]), timestamp=float(index))
    for index in range(10, 20):
        tracker.update(make_frame_result([make_face_prediction("angry", 0.1)]), timestamp=float(index))
    assert tracker.trend() < 0


def test_trend_is_zero_when_short(tracker):
    assert tracker.trend(window=10) == 0.0


def test_distress_alert_fires(labels, make_frame_result, make_face_prediction):
    config = AlertConfig(distress_share=0.5, cooldown_seconds=0.0, min_faces=2)
    session = SessionTracker(labels=labels, alert_config=config)
    session.update(
        make_frame_result(
            [
                make_face_prediction("angry", 0.1, box=(0, 0, 40, 40)),
                make_face_prediction("angry", 0.1, box=(80, 0, 40, 40)),
            ]
        ),
        timestamp=0.0,
    )
    assert any(alert.kind == "distress_cluster" for alert in session.alerts)


def test_alert_cooldown_suppresses_repeats(labels, make_frame_result, make_face_prediction):
    config = AlertConfig(distress_share=0.5, cooldown_seconds=30.0, min_faces=2)
    session = SessionTracker(labels=labels, alert_config=config)
    frame = make_frame_result(
        [
            make_face_prediction("sad", 0.2, box=(0, 0, 40, 40)),
            make_face_prediction("sad", 0.2, box=(80, 0, 40, 40)),
        ]
    )
    session.update(frame, timestamp=0.0)
    session.update(frame, timestamp=1.0)
    distress_alerts = [a for a in session.alerts if a.kind == "distress_cluster"]
    assert len(distress_alerts) == 1


def test_alert_requires_min_faces(labels, make_face_prediction, make_frame_result):
    config = AlertConfig(distress_share=0.5, min_faces=3)
    session = SessionTracker(labels=labels, alert_config=config)
    session.update(make_frame_result([make_face_prediction("angry", 0.1)]), timestamp=0.0)
    assert session.alerts == []


def test_low_engagement_alert(labels, make_face_prediction, make_frame_result):
    config = AlertConfig(low_engagement=0.4, low_engagement_share=0.5, cooldown_seconds=0.0, min_faces=2)
    session = SessionTracker(labels=labels, alert_config=config)
    session.update(
        make_frame_result(
            [
                make_face_prediction("angry", 0.05, box=(0, 0, 40, 40)),
                make_face_prediction("angry", 0.05, box=(80, 0, 40, 40)),
            ]
        ),
        timestamp=0.0,
    )
    assert any(alert.kind == "low_engagement" for alert in session.alerts)


def test_smoothed_engagement_short_series_passthrough(tracker, make_frame_result, make_face_prediction):
    tracker.update(make_frame_result([make_face_prediction("happy", 0.9)]), timestamp=0.0)
    assert len(tracker.smoothed_engagement()) == 1


def test_smoothed_engagement_smooths(tracker, make_frame_result, make_face_prediction):
    for index in range(10):
        tracker.update(make_frame_result([make_face_prediction("happy", 0.9)]), timestamp=float(index))
    smoothed = tracker.smoothed_engagement()
    assert len(smoothed) == 10
    assert all(0.0 <= value <= 1.0 for _, value in smoothed)


def test_reset_clears_state(tracker, make_frame_result, make_face_prediction):
    tracker.update(make_frame_result([make_face_prediction("happy", 0.9)]), timestamp=0.0)
    tracker.reset()
    report = tracker.report()
    assert report["frames_seen"] == 0
    assert report["students_detected"] == 0


def test_report_is_json_serializable(tracker, make_frame_result, make_face_prediction):
    tracker.update(make_frame_result([make_face_prediction("happy", 0.9)]), timestamp=0.0)
    json.dumps(tracker.report())


def test_many_frames_stay_on_one_track(tracker, make_frame_result, make_face_prediction):
    for _ in range(50):
        tracker.update(make_frame_result([make_face_prediction("happy", 0.9)]), timestamp=1.0)
    assert len(tracker.tracks) == 1


# --------------------------------------------------------------------------- #
# Tables and exports
# --------------------------------------------------------------------------- #


def test_build_student_table(tracker, make_frame_result, make_face_prediction):
    tracker.update(make_frame_result([make_face_prediction("happy", 0.9, box=(0, 0, 40, 40))]), timestamp=0.0)
    rows = build_student_table(tracker)
    assert rows[0]["Dominant emotion"] == "happy"
    assert "Engagement" in rows[0]


def test_build_student_table_empty(tracker):
    assert build_student_table(tracker) == []


def test_build_engagement_dataframe(tracker, make_frame_result, make_face_prediction):
    tracker.update(make_frame_result([make_face_prediction("happy", 0.9)]), timestamp=1.5)
    rows = build_engagement_dataframe(tracker)
    assert rows[0]["Time (s)"] == pytest.approx(1.5)


def test_export_session_csv(tracker, make_frame_result, make_face_prediction, tmp_path):
    tracker.update(make_frame_result([make_face_prediction("happy", 0.9, box=(0, 0, 40, 40))]), timestamp=0.0)
    path = export_session_csv(tracker, tmp_path / "out.csv")
    content = open(path, encoding="utf-8").read()
    assert "track_id" in content
    assert "happy" in content


def test_export_session_json(tracker, tmp_path):
    path = export_session_json(tracker, tmp_path / "out.json")
    payload = json.loads(open(path, encoding="utf-8").read())
    assert "frames_seen" in payload


def test_export_frame_csv_with_results(tmp_path, make_frame_result, make_face_prediction):
    result = make_frame_result([make_face_prediction("happy", 0.9)]).to_dict()
    result["source"] = "classroom.jpg"
    path = export_frame_csv([result], tmp_path / "frames.csv")
    content = open(path, encoding="utf-8").read()
    assert "classroom.jpg" in content
    assert "mean_engagement" in content


def test_export_frame_csv_empty(tmp_path):
    path = export_frame_csv([], tmp_path / "frames.csv")
    assert "source" in open(path, encoding="utf-8").read()


def test_export_frame_csv_counts_uncertain(tmp_path, make_face_prediction, make_frame_result):
    result = make_frame_result([make_face_prediction("happy", certain=False)]).to_dict()
    result["source"] = "a.jpg"
    path = export_frame_csv([result], tmp_path / "frames.csv")
    assert "uncertain_faces" in open(path, encoding="utf-8").read()


# --------------------------------------------------------------------------- #
# Heatmap
# --------------------------------------------------------------------------- #


def test_heatmap_matrix_rows_sum_to_one(tracker, labels, make_frame_result, make_face_prediction):
    for index in range(12):
        tracker.update(
            make_frame_result(
                [
                    make_face_prediction("happy", 0.9, box=(0, 0, 40, 40)),
                    make_face_prediction("sad", 0.3, box=(90, 0, 40, 40)),
                ]
            ),
            timestamp=float(index),
        )
    matrix = heatmap_matrix(tracker, labels, buckets=4)
    assert len(matrix) == 4
    assert all(len(row) == len(labels) for row in matrix)
    assert all(sum(row) == pytest.approx(1.0) for row in matrix)


def test_heatmap_matrix_empty(tracker, labels):
    assert heatmap_matrix(tracker, labels) == []


def test_heatmap_matrix_no_labels(tracker):
    assert heatmap_matrix(tracker, []) == []


def test_heatmap_frame_shape(tracker, labels, make_frame_result, make_face_prediction):
    tracker.update(make_frame_result([make_face_prediction("happy", 0.9)]), timestamp=0.0)
    frame = heatmap_frame(heatmap_matrix(tracker, labels, buckets=2), labels)
    assert frame
    assert set(frame[0]) == {"bucket", *labels}
