import shutil
import subprocess

import pytest

from face_rec.sources import GStreamerSource, open_source

PY = "/usr/bin/python3"


def _has_gst() -> bool:
    try:
        r = subprocess.run([PY, "-c", "import gi; gi.require_version('Gst', '1.0')"],
                           capture_output=True, timeout=20)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(not _has_gst(), reason="system Python lacks GStreamer bindings")
OPTS = {"backend": "gstreamer", "gstreamer": {"python": PY}, "open_timeout_ms": 15000}


@pytest.fixture(scope="module")
def h264_file(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available")
    path = tmp_path_factory.mktemp("v") / "clip.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=320x240:rate=10:duration=2", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return str(path)


def test_custom_pipeline_live():
    with open_source("videotestsrc is-live=true ! video/x-raw,width=64,height=48,framerate=30/1"
                     " ! videoconvert",
                     OPTS) as src:
        assert isinstance(src, GStreamerSource) and src.is_live
        frames = [f for _, f in zip(range(5), src)]
    assert frames[0].image.shape == (48, 64, 3)


@pytest.mark.parametrize("decoder", ["avdec_h264", "qtic2vdec"])
def test_file_every_frame(h264_file, decoder):
    opts = {**OPTS, "gstreamer": {**OPTS["gstreamer"], "decoder": decoder, "convert": "videoconvert"}}
    try:
        src = open_source(h264_file, opts)
    except ConnectionError as e:
        pytest.skip(f"{decoder} unavailable: {e}")
    with src:
        assert not src.is_live and src.size == (320, 240) and src.fps == 10
        frames = list(src)
    assert len(frames) == 20
    assert frames[-1].timestamp > frames[0].timestamp


def test_bad_pipeline_raises():
    with pytest.raises(ConnectionError):
        open_source("this_element_does_not_exist ! fakesink", {**OPTS, "open_timeout_ms": 5000})
