"""Tests for the camera bridge, frame buffer, and image discovery."""

from __future__ import annotations

import json
import threading
import urllib.request

import pytest

pytest.importorskip("numpy")

from src.camera import (
    WEBAPP_HTML,
    CameraBridge,
    FrameBuffer,
    list_images,
    load_batch,
)


# --------------------------------------------------------------------------- #
# FrameBuffer
# --------------------------------------------------------------------------- #


def test_buffer_starts_empty():
    buffer = FrameBuffer()
    assert len(buffer) == 0
    assert buffer.latest() is None


def test_buffer_push_and_latest():
    buffer = FrameBuffer()
    assert buffer.push(b"first") == 1
    assert buffer.push(b"second") == 2
    seq, data, _ = buffer.latest()
    assert (seq, data) == (2, b"second")


def test_buffer_ignores_empty_payloads():
    buffer = FrameBuffer()
    assert buffer.push(b"") == 0
    assert len(buffer) == 0


def test_buffer_respects_capacity():
    buffer = FrameBuffer(capacity=2)
    for payload in (b"a", b"b", b"c"):
        buffer.push(payload)
    assert len(buffer) == 2
    assert buffer.latest()[1] == b"c"


def test_buffer_recent_order():
    buffer = FrameBuffer(capacity=5)
    for payload in (b"a", b"b", b"c"):
        buffer.push(payload)
    assert [data for _, data, _ in buffer.recent()] == [b"a", b"b", b"c"]
    assert [data for _, data, _ in buffer.recent(2)] == [b"b", b"c"]


def test_buffer_clear():
    buffer = FrameBuffer()
    buffer.push(b"x")
    buffer.clear()
    assert len(buffer) == 0
    assert buffer.push(b"y") == 1


def test_buffer_is_thread_safe():
    buffer = FrameBuffer(capacity=100)
    threads = [
        threading.Thread(target=lambda i=i: [buffer.push(bytes([i % 251])) for _ in range(20)])
        for i in range(5)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert buffer.latest()[0] == 100


# --------------------------------------------------------------------------- #
# CameraBridge
# --------------------------------------------------------------------------- #


@pytest.fixture
def bridge():
    server = CameraBridge(port=0)
    server.start()
    yield server
    server.stop()


def test_bridge_binds_ephemeral_port(bridge):
    assert bridge.running
    assert bridge.port > 0
    assert bridge.url.startswith("http://127.0.0.1:")


def test_bridge_serves_webapp(bridge):
    with urllib.request.urlopen(bridge.url, timeout=5) as response:
        body = response.read().decode("utf-8")
    assert response.status == 200
    assert body == WEBAPP_HTML


def test_bridge_accepts_frame(bridge):
    request = urllib.request.Request(
        bridge.url + "frame", data=b"\xff\xd8\xff\xd9", method="POST"
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    assert payload["seq"] == 1
    assert bridge.buffer.latest()[1] == b"\xff\xd8\xff\xd9"


def test_bridge_latest_endpoint(bridge):
    urllib.request.urlopen(
        urllib.request.Request(bridge.url + "frame", data=b"jpegdata", method="POST"), timeout=5
    ).read()
    with urllib.request.urlopen(bridge.url + "latest", timeout=5) as response:
        assert response.read() == b"jpegdata"


def test_bridge_rejects_empty_post(bridge):
    request = urllib.request.Request(bridge.url + "frame", data=b"", method="POST")
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=5)
    assert excinfo.value.code == 400


def test_bridge_stats_endpoint(bridge):
    with urllib.request.urlopen(bridge.url + "stats", timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    assert payload["buffered"] == 0


def test_bridge_latest_404_when_empty(bridge):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(bridge.url + "latest", timeout=5)
    assert excinfo.value.code == 404


def test_bridge_stop_clears_buffer():
    server = CameraBridge(port=0)
    server.start()
    server.buffer.push(b"x")
    server.stop()
    assert not server.running
    assert len(server.buffer) == 0


def test_bridge_context_manager():
    with CameraBridge(port=0) as server:
        assert server.running
        assert server.port > 0
    assert not server.running


def test_bridge_start_is_idempotent():
    server = CameraBridge(port=0)
    first = server.start()
    assert server.start() == first
    server.stop()


def test_webapp_html_uses_getusermedia():
    assert "getUserMedia" in WEBAPP_HTML
    assert "/frame" in WEBAPP_HTML


# --------------------------------------------------------------------------- #
# Image discovery
# --------------------------------------------------------------------------- #


def test_list_images_from_file(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    path = tmp_path / "a.png"
    Image.new("RGB", (10, 10)).save(path)
    assert list_images(path) == [path]


def test_list_images_rejects_non_image_file(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")
    assert list_images(path) == []


def test_list_images_from_directory(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    for name in ("b.png", "a.jpg", "c.txt"):
        target = tmp_path / name
        if name.endswith(".txt"):
            target.write_text("x", encoding="utf-8")
        else:
            Image.new("RGB", (10, 10)).save(target)

    found = [p.name for p in list_images(tmp_path)]
    assert found == ["a.jpg", "b.png"]


def test_list_images_missing_directory(tmp_path):
    assert list_images(tmp_path / "absent") == []


def test_load_batch_clamps():
    assert len(load_batch([f"a{i}.png" for i in range(50)], max_images=10)) == 10
    assert load_batch([], max_images=5) == []
