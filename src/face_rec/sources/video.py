"""Frame sources: video files and live streams (RTSP/RTMP/HTTP, camera index).

Files are read sequentially, every frame, as fast as processing allows.
Live sources are read on a background thread that keeps only the newest frame,
so slow processing drops stale frames instead of building up latency, and the
stream is reopened automatically when it stalls or disconnects.

Backends (`source.backend` in the config):
    opencv     OpenCV + FFmpeg, software decode (default; works everywhere)
    gstreamer  GStreamer pipeline, e.g. Qualcomm hardware decode (qtic2vdec) and
               color conversion (qtivtransform), run in a helper process
"""

from __future__ import annotations

import logging
import os
import select
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)

LIVE_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "tcp://")


@dataclass
class Frame:
    image: np.ndarray  # BGR uint8
    index: int         # frames read from the source (gaps = dropped frames on live sources)
    timestamp: float   # seconds: position in file, or wall-clock time for live sources


def open_source(source: str | int, options: dict[str, Any] | None = None) -> "BaseSource":
    """Create the source for the configured backend (default: opencv)."""
    options = options or {}
    backend = options.get("backend", "opencv")
    if backend == "opencv":
        return VideoSource(source, options)
    if backend == "gstreamer":
        return GStreamerSource(source, options)
    raise ValueError(f"Unknown source backend '{backend}' (opencv | gstreamer)")


class BaseSource:
    """Shared iteration, live reader thread and reconnect logic.

    Options (the `source:` section of the config):
        open_timeout_ms:     10000
        read_timeout_ms:     5000
        reconnect:           true   (live only)
        reconnect_delay:     1.0    seconds, doubles up to max_reconnect_delay
        max_reconnect_delay: 30.0
        max_reconnects:      0      0 = retry forever
        loop:                false  (files only) restart from the beginning at EOF

    Backends implement `_open`, `_read`, `_release` and the `fps` / `size` properties.
    """

    def __init__(self, source: str | int, options: dict[str, Any] | None = None):
        self.options = options or {}
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.source = source
        self.is_live = self._is_live(source)
        if not self.is_live and not Path(source).exists():
            raise FileNotFoundError(f"Video file not found: {source}")

        self.reconnects = 0
        self.dropped = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._error: Exception | None = None
        self._index = 0

        self._open()
        if self.is_live:
            self._thread = threading.Thread(target=self._reader, name="video-reader", daemon=True)
            self._thread.start()

    # ---- backend API ----
    @staticmethod
    def _is_live(source) -> bool:
        return isinstance(source, int) or str(source).lower().startswith(LIVE_SCHEMES)

    def _open(self) -> None:
        """Open the source; raise ConnectionError on failure."""
        raise NotImplementedError

    def _read(self) -> tuple[bool, np.ndarray | None, float | None]:
        """Next frame: (ok, BGR image, position in seconds or None)."""
        raise NotImplementedError

    def _release(self) -> None:
        raise NotImplementedError

    def _rewind(self) -> None:
        """Restart a file from the beginning (used by `loop`)."""
        self._release()
        self._open()

    @property
    def fps(self) -> float:
        raise NotImplementedError

    @property
    def size(self) -> tuple[int, int]:
        raise NotImplementedError

    @property
    def frame_count(self) -> int:
        """Total frames for files (0 if unknown / live)."""
        return 0

    @property
    def description(self) -> str:
        return _redact(str(self.source))

    # ---- reconnect / close ----
    def _reopen_with_backoff(self) -> bool:
        """Reopen a live source; returns False if stopped or out of retries."""
        delay = float(self.options.get("reconnect_delay", 1.0))
        max_delay = float(self.options.get("max_reconnect_delay", 30.0))
        max_tries = int(self.options.get("max_reconnects", 0))
        self._release()
        while not self._stop.is_set():
            if max_tries and self.reconnects >= max_tries:
                return False
            self.reconnects += 1
            log.warning("Reconnecting to %s in %.1fs (attempt %d)",
                        self.description, delay, self.reconnects)
            if self._stop.wait(delay):
                return False
            try:
                self._open()
                return True
            except ConnectionError as e:
                log.warning("%s", e)
                delay = min(delay * 2, max_delay)
        return False

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- live reader thread ----
    def _reader(self) -> None:
        reconnect = bool(self.options.get("reconnect", True))
        while not self._stop.is_set():
            ok, img, _ = self._read()
            if not ok:
                if self._stop.is_set():
                    return
                if not reconnect or not self._reopen_with_backoff():
                    with self._cond:
                        if not self._stop.is_set():
                            self._error = ConnectionError(f"Stream ended: {self.description}")
                        self._stop.set()
                        self._cond.notify_all()
                    return
                with self._cond:
                    self._latest = None  # drop the stale pre-disconnect frame
                continue
            frame = Frame(img, self._index, time.time())
            self._index += 1
            with self._cond:
                if self._latest is not None:
                    self.dropped += 1
                self._latest = frame
                self._cond.notify()

    # ---- iteration ----
    def __iter__(self) -> Iterator[Frame]:
        return self._iter_live() if self.is_live else self._iter_file()

    def _iter_file(self) -> Iterator[Frame]:
        loop = bool(self.options.get("loop", False))
        while not self._stop.is_set():
            ok, img, pos = self._read()
            if not ok:
                if loop and self._index > 0:
                    self._rewind()
                    continue
                return
            yield Frame(img, self._index, pos if pos is not None else self._index / self.fps)
            self._index += 1

    def _iter_live(self) -> Iterator[Frame]:
        while True:
            with self._cond:
                while self._latest is None and not self._stop.is_set():
                    self._cond.wait(timeout=1.0)
                if self._latest is None:  # stopped
                    if self._error is not None:
                        raise self._error
                    return
                frame, self._latest = self._latest, None
            yield frame


class VideoSource(BaseSource):
    """OpenCV + FFmpeg backend (software decode).

    Extra options: rtsp_transport: tcp (default, robust) | udp (lower latency)
    """

    cap: cv2.VideoCapture | None = None

    def _open(self) -> None:
        opts = self.options
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(opts.get("open_timeout_ms", 10000)),
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(opts.get("read_timeout_ms", 5000)),
        ]
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source, cv2.CAP_ANY, params)
        else:
            if str(self.source).lower().startswith(("rtsp://", "rtsps://")):
                # Read by OpenCV's FFmpeg backend when the capture is created.
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    f"rtsp_transport;{opts.get('rtsp_transport', 'tcp')}"
                )
            cap = cv2.VideoCapture(str(self.source), cv2.CAP_FFMPEG, params)
        if not cap.isOpened():
            cap.release()
            raise ConnectionError(f"Cannot open video source: {self.description}")
        if self.is_live:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        w, h = self.size
        log.info("Opened %s (%dx%d @ %.1f fps, %s, opencv)", self.description, w, h, self.fps,
                 "live" if self.is_live else f"{self.frame_count} frames")

    def _read(self):
        if self.cap is None:
            return False, None, None
        ok, img = self.cap.read()
        return ok, img, (self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if ok else None)

    def _release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _rewind(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    @property
    def fps(self) -> float:
        fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 0.0
        return fps if 0 < fps < 240 else 25.0

    @property
    def size(self) -> tuple[int, int]:
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    @property
    def frame_count(self) -> int:
        return 0 if self.is_live else int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))


_HEADER = struct.Struct("<4sB3xIIIIIqd")  # must match gst_helper.HEADER
_FMT_BGR, _FMT_NV12 = 0, 1
_HELPER = Path(__file__).with_name("gst_helper.py")


class GStreamerSource(BaseSource):
    """GStreamer backend, run in a helper process under the system Python (which has the
    GStreamer bindings for the vendor plugins); frames come back over a pipe.

    Extra options, under `source.gstreamer`:
        python:   interpreter with PyGObject + GStreamer (default /usr/bin/python3)
        codec:    h264 | h265                       (RTSP/file templates)
        decoder:  qtic2vdec (Qualcomm HW, default) | avdec_h264 (CPU) | ...
        format:   nv12 (default): decoder output as-is, converted to BGR here with OpenCV,
                  only for frames that are delivered | bgr: converted in the pipeline
        convert:  element(s) between decoder and sink; default none for qtic2vdec + nv12,
                  else videoconvert ("qtivtransform engine=fcv" also works; engine=gles
                  hangs on this box)
        latency:  rtspsrc jitter buffer in ms (default 100)
        max_fps:  >0: the helper drops frames before copying them, so only N frames/s
                  cross the pipe and get converted; set it to tracking.active_fps when no
                  output video is written (0 = all frames)
        width, height: scale in the pipeline (needs a scaling `convert`; 0 = keep size)
        pipeline: full custom pipeline (without sink); {source} is replaced by the URL/path
    A source containing " ! " is used as the pipeline as-is.
    """

    proc: subprocess.Popen | None = None

    @staticmethod
    def _is_live(source) -> bool:
        return BaseSource._is_live(source) or " ! " in str(source)

    @property
    def _g(self) -> dict[str, Any]:
        return self.options.get("gstreamer", {}) or {}

    def _pipeline(self) -> str:
        g, src = self._g, str(self.source)
        if " ! " in src:
            return src
        if g.get("pipeline"):
            return g["pipeline"].format(source=src)
        if isinstance(self.source, int):
            raise ValueError("camera index is not supported by the gstreamer backend; "
                             "pass a pipeline, e.g. 'qtiqmmfsrc ! video/x-raw,width=1280,height=720'")
        codec = g.get("codec", "h264")
        decoder = g.get("decoder", "qtic2vdec")
        native = decoder.startswith("qtic2vdec") and g.get("format", "nv12").lower() == "nv12"
        convert = g.get("convert", "" if native else "videoconvert")
        if src.lower().startswith(("rtsp://", "rtsps://")):
            transport = self.options.get("rtsp_transport", "tcp")
            head = (f'rtspsrc location="{src}" latency={int(g.get("latency", 100))} '
                    f"protocols={transport} ! rtp{codec}depay")
        else:
            demux = "matroskademux" if src.lower().endswith(".mkv") else "qtdemux"
            head = f'filesrc location="{src}" ! {demux}'
        return f"{head} ! {codec}parse ! {decoder}" + (f" ! {convert}" if convert else "")

    def _open(self) -> None:
        g = self._g
        cmd = [g.get("python", "/usr/bin/python3"), "-I", str(_HELPER), self._pipeline(),
               "--format", str(g.get("format", "nv12")).upper(),
               "--timeout", str(int(self.options.get("read_timeout_ms", 5000)) / 1000)]
        if self.is_live:
            cmd.append("--live")
        if float(g.get("max_fps", 0) or 0) > 0:
            cmd += ["--max-fps", str(float(g["max_fps"]))]
        if g.get("width") and g.get("height"):
            cmd += ["--width", str(int(g["width"])), "--height", str(int(g["height"]))]
        log.debug("gst pipeline: %s", cmd[3])
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._pending = None
        self._info = (0, 0, 0.0)
        # Wait for the first frame so size/fps are known and failures surface here.
        timeout = int(self.options.get("open_timeout_ms", 10000)) / 1000
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        first = self._read_frame() if ready else None
        if first is None:
            err = self._stop_helper()
            raise ConnectionError(f"Cannot open video source: {self.description}"
                                  + (f" ({err})" if err else " (timeout)"))
        self._pending = first
        w, h = self.size
        chain = cmd[3].split(" ! ", 2)[-1] if " ! " in cmd[3] else cmd[3]
        log.info("Opened %s (%dx%d @ %.1f fps, %s, gstreamer: %s -> %s%s)", self.description,
                 w, h, self.fps, "live" if self.is_live else "file", chain, cmd[5],
                 f", max {g['max_fps']} fps" if "--max-fps" in cmd else "")

    def _read_frame(self):
        out = self.proc.stdout if self.proc else None
        if out is None:
            return None
        head = out.read(_HEADER.size)
        if len(head) < _HEADER.size:
            return None
        magic, fmt, w, h, s0, s1, size, pts, fps = _HEADER.unpack(head)
        if magic != b"FRM2":
            raise RuntimeError("gst_helper protocol error")
        data = out.read(size)
        if len(data) < size:
            return None
        buf = np.frombuffer(data, np.uint8)
        if fmt == _FMT_NV12:
            y = buf[: s0 * h].reshape(h, s0)[:, :w]
            uv = buf[s0 * h:].reshape(h // 2, s1)[:, :w].reshape(h // 2, w // 2, 2)
            img = cv2.cvtColorTwoPlane(y, uv, cv2.COLOR_YUV2BGR_NV12)
        else:
            img = buf.reshape(h, s0)[:, : w * 3].reshape(h, w, 3)
        self._info = (w, h, fps)
        return img, (pts / 1e9 if pts >= 0 else None)

    def _read(self):
        if self._pending is not None:
            (img, pos), self._pending = self._pending, None
            return True, img, pos
        got = self._read_frame()
        if got is None:
            err = self._stop_helper()
            if err and not self._stop.is_set():
                log.warning("gstreamer: %s", err)
            return False, None, None
        return True, got[0], got[1]

    def _stop_helper(self) -> str:
        """Terminate the helper; return the tail of its stderr."""
        p, self.proc = self.proc, None
        if p is None:
            return ""
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        try:
            err = p.stderr.read().decode(errors="replace").strip().splitlines()
        except Exception:
            err = []
        msgs = [line for line in err if line.startswith("gst_helper")] or err[-1:]
        return msgs[-1] if msgs else ""

    def _release(self) -> None:
        self._stop_helper()

    def close(self) -> None:
        # Stop the helper first: the reader thread is blocked on the pipe and returns on EOF.
        self._stop.set()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        super().close()

    @property
    def fps(self) -> float:
        fps = self._info[2]
        return fps if 0 < fps < 240 else 25.0

    @property
    def size(self) -> tuple[int, int]:
        return self._info[0], self._info[1]


def _redact(url: str) -> str:
    """Hide credentials in rtsp://user:pass@host URLs for logging."""
    if "://" in url and "@" in url.split("://", 1)[1].split("/", 1)[0]:
        scheme, rest = url.split("://", 1)
        return f"{scheme}://***@{rest.split('@', 1)[1]}"
    return url
