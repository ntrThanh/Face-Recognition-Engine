import cv2
import numpy as np
import pytest

from face_rec.sources import VideoSource
from face_rec.sources.video import _redact


@pytest.fixture
def video(tmp_path):
    path = str(tmp_path / "v.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(12):
        w.write(np.full((48, 64, 3), i * 20, np.uint8))
    w.release()
    return path


def test_reads_every_frame_of_a_file(video):
    with VideoSource(video) as src:
        assert not src.is_live and src.size == (64, 48) and src.fps == 10
        frames = list(src)
    assert [f.index for f in frames] == list(range(12))
    assert frames[-1].timestamp > frames[0].timestamp


def test_loop_restarts_file(video):
    with VideoSource(video, {"loop": True}) as src:
        frames = [f for _, f in zip(range(30), src)]
    assert len(frames) == 30


def test_missing_file_and_dead_stream():
    with pytest.raises(FileNotFoundError):
        VideoSource("/nonexistent.mp4")
    with pytest.raises(ConnectionError):
        VideoSource("rtsp://127.0.0.1:1/none", {"open_timeout_ms": 2000})


def test_redacts_credentials():
    assert _redact("rtsp://admin:secret@10.0.0.2:554/s1") == "rtsp://***@10.0.0.2:554/s1"
    assert _redact("rtsp://10.0.0.2/s1") == "rtsp://10.0.0.2/s1"
