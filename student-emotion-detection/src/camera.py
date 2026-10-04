"""Camera sources: webcam, video files, and image directories.

Streamlit cannot hold an open ``cv2.VideoCapture`` across reruns reliably, so
the webcam path is built around a small Flask-free stdlib HTTP bridge: the
browser grabs frames with ``getUserMedia``, POSTs JPEG bytes to the local
bridge, and the app analyses them on the next rerun. That keeps the whole
stack dependency-light and works over Streamlit's iframe sandbox.

Video and image-directory sources use OpenCV directly and are used by the CLI
scripts and the tests.
"""

from __future__ import annotations

import base64
import json
import threading
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from src.config import VIDEO_SAMPLE_STRIDE, WEBCAM_MAX_SIDE

try:  # pragma: no cover - import guard
    import numpy as np

    NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

try:  # pragma: no cover - import guard
    import cv2

    CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Video files
# --------------------------------------------------------------------------- #


@dataclass
class VideoSample:
    """One analysed video frame."""

    frame_index: int
    timestamp: float
    face_count: int
    mean_engagement: float
    dominant_label: str
    histogram: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "face_count": self.face_count,
            "mean_engagement": self.mean_engagement,
            "dominant_label": self.dominant_label,
            "histogram": dict(self.histogram),
        }


@dataclass
class VideoAnalysis:
    """Aggregate results from analysing a whole video file."""

    path: str
    fps: float
    frame_count: int
    frames_analysed: int
    duration_seconds: float
    samples: List[VideoSample] = field(default_factory=list)

    def engagement_curve(self) -> List[Tuple[float, float]]:
        """``(timestamp, engagement)`` pairs for plotting."""

        return [(s.timestamp, s.mean_engagement) for s in self.samples]

    def summary(self) -> Dict[str, Any]:
        """Headline numbers for the UI summary card."""

        engagements = [s.mean_engagement for s in self.samples]
        labels: Dict[str, int] = {}
        for sample in self.samples:
            for label, count in sample.histogram.items():
                labels[label] = labels.get(label, 0) + count
        return {
            "path": self.path,
            "fps": round(self.fps, 2),
            "frames_analysed": self.frames_analysed,
            "duration_seconds": round(self.duration_seconds, 2),
            "mean_engagement": (sum(engagements) / len(engagements)) if engagements else 0.0,
            "peak_engagement": max(engagements) if engagements else 0.0,
            "trough_engagement": min(engagements) if engagements else 0.0,
            "avg_faces": (
                sum(s.face_count for s in self.samples) / len(self.samples)
            ) if self.samples else 0.0,
            "label_totals": labels,
            "dominant_label": (
                max(sorted(labels.items()), key=lambda kv: kv[1])[0] if labels else "neutral"
            ),
        }


def analyse_video(
    path: Any,
    predictor: Any,
    stride: int = VIDEO_SAMPLE_STRIDE,
    max_samples: int = 300,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> VideoAnalysis:
    """Analyse a video file frame-by-frame with a sampling ``stride``.

    Sampling every ``stride``-th frame keeps a 10-minute lecture tractable while
    still covering the whole timeline.
    """

    if not CV2_AVAILABLE:
        raise ImportError("opencv-python is required for video analysis.")

    path = str(path)
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(stride))

    samples: List[VideoSample] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0 and len(samples) < max_samples:
            result = predictor.predict(frame, apply_smoothing=False)
            samples.append(
                VideoSample(
                    frame_index=index,
                    timestamp=index / fps,
                    face_count=result.face_count,
                    mean_engagement=result.mean_engagement,
                    dominant_label=result.dominant_label,
                    histogram=result.label_histogram(),
                )
            )
            if progress_callback is not None:
                progress_callback(index, total_frames)
        index += 1
        if len(samples) >= max_samples:
            break

    capture.release()
    duration = (index / fps) if fps else 0.0
    return VideoAnalysis(
        path=path,
        fps=fps,
        frame_count=index,
        frames_analysed=len(samples),
        duration_seconds=duration,
        samples=samples,
    )


def resize_for_display(frame: "np.ndarray", max_side: int = WEBCAM_MAX_SIDE) -> "np.ndarray":
    """Downscale a frame so its longest side is at most ``max_side``."""

    if frame is None:
        return frame
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_side or longest == 0:
        return frame
    scale = max_side / float(longest)
    new_size = (int(width * scale), int(height * scale))
    if CV2_AVAILABLE:
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    return np.asarray(  # pragma: no cover - only without opencv
        [
            [
                [
                    frame[int(y / scale), int(x / scale)][c]
                    for x in range(new_size[0])
                ]
                for y in range(new_size[1])
            ]
            for c in range(3)
        ],
        dtype=frame.dtype,
    )


# --------------------------------------------------------------------------- #
# Webcam bridge
# --------------------------------------------------------------------------- #

#: Frames are kept in a short deque so a slow rerun drops old frames instead of
#: stalling the browser.
FRAME_BUFFER_SIZE = 3


class FrameBuffer:
    """Thread-safe ring buffer of the most recent camera frames.

    The browser thread writes JPEG bytes; the Streamlit thread reads them. A
    monotonically increasing ``seq`` lets the UI skip frames it already analysed.
    """

    def __init__(self, capacity: int = FRAME_BUFFER_SIZE) -> None:
        self.capacity = max(1, int(capacity))
        self._frames: Deque[Tuple[int, bytes, float]] = deque(maxlen=self.capacity)
        self._lock = threading.Lock()
        self._seq = 0

    def push(self, data: bytes) -> int:
        """Store a frame and return its sequence number."""

        if not data:
            return self._seq
        import time

        with self._lock:
            self._seq += 1
            self._frames.append((self._seq, bytes(data), time.time()))
            return self._seq

    def latest(self) -> Optional[Tuple[int, bytes, float]]:
        """Newest ``(seq, jpeg_bytes, timestamp)``, or ``None`` when empty."""

        with self._lock:
            if not self._frames:
                return None
            return self._frames[-1]

    def recent(self, count: Optional[int] = None) -> List[Tuple[int, bytes, float]]:
        """Oldest-to-newest list of buffered frames."""

        with self._lock:
            items = list(self._frames)
        return items[-count:] if count else items

    def clear(self) -> None:
        """Drop all buffered frames and reset the sequence counter."""

        with self._lock:
            self._frames.clear()
            self._seq = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)


WEBAPP_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  body { margin: 0; background: #111; color: #eee;
         font-family: -apple-system, Segoe UI, Roboto, sans-serif; }
  #wrap { display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 10px; }
  video, canvas { max-width: 100%; border-radius: 8px; }
  canvas { display: none; }
  button { background: #1f6feb; color: #fff; border: 0; border-radius: 6px;
           padding: 8px 16px; font-size: 14px; cursor: pointer; }
  button.stop { background: #b3261e; }
  #status { font-size: 12px; opacity: 0.8; }
</style>
</head>
<body>
<div id="wrap">
  <video id="video" autoplay playsinline muted></video>
  <canvas id="canvas"></canvas>
  <div>
    <button id="start">Start camera</button>
    <button id="stop" class="stop" disabled>Stop</button>
  </div>
  <div id="status">idle</div>
</div>
<script>
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('start');
const stopBtn = document.getElementById('stop');
const status = document.getElementById('status');
let stream = null, timer = null, sending = false;

function setStatus(text) { status.textContent = text; }

async function start() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = stream;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    setStatus('streaming');
    timer = setInterval(grab, 400);
  } catch (err) {
    setStatus('camera error: ' + err.message);
  }
}

async function grab() {
  if (sending || !stream || video.readyState < 2) return;
  sending = true;
  try {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.85));
    const response = await fetch('/frame', { method: 'POST', body: blob });
    const info = await response.json();
    setStatus('streaming - seq ' + info.seq);
  } catch (err) {
    setStatus('send error: ' + err.message);
  } finally {
    sending = false;
  }
}

function stop() {
  if (timer) { clearInterval(timer); timer = null; }
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus('stopped');
}

startBtn.addEventListener('click', start);
stopBtn.addEventListener('click', stop);
window.addEventListener('beforeunload', stop);
</script>
</body>
</html>
"""


class CameraBridge:
    """Tiny HTTP server that receives webcam frames from the browser.

    Runs on localhost only, on an ephemeral port by default. Intended as a
    local development / demo aid - it is not hardened for exposure on a shared
    network, and the README says so.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.buffer = FrameBuffer()
        self.host = host
        self.requested_port = int(port)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        """Actual bound port (resolved after :meth:`start`)."""

        if self._server is None:
            return self.requested_port
        return int(self._server.server_address[1])

    @property
    def running(self) -> bool:
        """True while the bridge is serving."""

        return self._server is not None

    def start(self) -> str:
        """Start serving in a daemon thread and return the base URL."""

        if self._server is not None:
            return self.url

        buffer = self.buffer

        class Handler(BaseHTTPRequestHandler):
            """Route for ``/`` (page) and ``/frame`` (JPEG upload)."""

            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
                """Silence the default stderr access log."""

            def _send(self, code: int, body: bytes, content_type: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/frame") or self.path.startswith("/stats"):
                    payload = json.dumps(
                        {
                            "buffered": len(buffer),
                            "seq": buffer.latest()[0] if buffer.latest() else 0,
                        }
                    ).encode("utf-8")
                    self._send(200, payload, "application/json")
                elif self.path.startswith("/latest"):
                    latest = buffer.latest()
                    if latest is None:
                        self._send(404, b"{}", "application/json")
                    else:
                        self._send(200, latest[1], "image/jpeg")
                else:
                    self._send(200, WEBAPP_HTML.encode("utf-8"), "text/html; charset=utf-8")

            def do_POST(self) -> None:  # noqa: N802
                if not self.path.startswith("/frame"):
                    self._send(404, b"{}", "application/json")
                    return
                length = int(self.headers.get("Content-Length") or 0)
                payload = self.rfile.read(length) if length else b""
                if not payload:
                    self._send(400, b'{"error":"empty body"}', "application/json")
                    return
                seq = buffer.push(payload)
                self._send(200, json.dumps({"seq": seq}).encode("utf-8"), "application/json")

        self._server = ThreadingHTTPServer((self.host, self.requested_port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        )
        self._thread.start()
        return self.url

    @property
    def url(self) -> str:
        """Base URL the browser should open."""

        return f"http://{self.host}:{self.port}/"

    def stop(self) -> None:
        """Shut the server down and clear buffered frames."""

        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self.buffer.clear()

    def __enter__(self) -> "CameraBridge":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


_BRIDGE: Optional[CameraBridge] = None


def get_camera_bridge() -> CameraBridge:
    """Return the process-wide bridge, starting it on first use.

    Streamlit reruns in the same process, so a module-level singleton avoids
    binding a new port on every rerun.
    """

    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = CameraBridge()
        _BRIDGE.start()
    return _BRIDGE


def stop_camera_bridge() -> None:
    """Stop the singleton bridge if it was started."""

    global _BRIDGE
    if _BRIDGE is not None:
        _BRIDGE.stop()
        _BRIDGE = None


# --------------------------------------------------------------------------- #
# Image sources
# --------------------------------------------------------------------------- #


def list_images(path: Any, recursive: bool = True) -> List[Path]:
    """List image files under ``path`` in a stable, sorted order."""

    path = Path(path)
    if path.is_file():
        return [path] if path.suffix.lower() in _IMAGE_SUFFIXES else []
    if not path.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    found = [
        p
        for p in sorted(path.glob(pattern))
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    ]
    return found


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def load_batch(paths: Sequence[Path], max_images: int = 200) -> List[Path]:
    """Clamp a path list to ``max_images`` so a folder dump cannot hang the UI."""

    return list(paths)[: max(0, int(max_images))]
