"""GStreamer frame pump, run as a subprocess by GStreamerSource.

It runs under the *system* Python, which has the PyGObject/GStreamer bindings for the
vendor GStreamer build (Qualcomm plugins), while the app runs in its own environment.
It must not import face_rec.

    /usr/bin/python3 -I gst_helper.py "<pipeline without sink>" [--format NV12|BGR]
        [--live] [--max-fps N] [--width W --height H] [--timeout S]

Frames go to the original stdout as HEADER + payload:
    HEADER = <4s "FRM2", uint8 format (0 BGR, 1 NV12), 3 pad, uint32 width, height,
              stride0, stride1, payload_size, int64 pts_ns (-1 if none), float64 fps>
    BGR  payload: height rows of stride0 bytes
    NV12 payload: Y (height rows of stride0) then UV (height/2 rows of stride1)
Plane layout comes from the buffer's VideoMeta: hardware decoders pad strides and
plane offsets (e.g. 480-wide NV12 with stride 512), which the caps do not show.

--max-fps drops frames here, before they are mapped or copied, so only frames the
app can use cross the pipe. Exit codes: 0 EOS, 2 pipeline error, 3 stalled, 4 bad args.
"""

import argparse
import os
import struct
import sys
import time

HEADER = struct.Struct("<4sB3xIIIIIqd")
FORMATS = {"BGR": 0, "NV12": 1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pipeline")
    ap.add_argument("--format", default="NV12", choices=sorted(FORMATS))
    ap.add_argument("--live", action="store_true", help="keep only the newest frame")
    ap.add_argument("--max-fps", type=float, default=0.0)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds without a frame")
    args = ap.parse_args()

    # Frames get a private copy of stdout; fd 1 is pointed at stderr so vendor libraries
    # that print to stdout (e.g. the GBM mapper) cannot corrupt the frame stream.
    out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)

    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstVideo", "1.0")
    gi.require_version("GstApp", "1.0")
    from gi.repository import Gst, GstApp, GstVideo  # noqa: F401  (GstApp adds AppSink methods)

    Gst.init(None)
    caps = f"video/x-raw,format={args.format}"
    if args.width and args.height:
        caps += f",width={args.width},height={args.height}"
    # live: drop stale frames inside GStreamer; file: block so no frame is lost
    sink = (f"appsink name=sink sync=false emit-signals=false "
            f"max-buffers={1 if args.live else 4} drop={'true' if args.live else 'false'}")
    try:
        pipe = Gst.parse_launch(f"{args.pipeline} ! {caps} ! {sink}")
    except Exception as e:  # GLib.Error on a bad description
        print(f"gst_helper: invalid pipeline: {e}", file=sys.stderr)
        return 4
    appsink = pipe.get_by_name("sink")
    bus = pipe.get_bus()
    fmt = FORMATS[args.format]
    interval = 1.0 / args.max_fps if args.max_fps > 0 else 0.0
    pipe.set_state(Gst.State.PLAYING)

    last_frame = time.monotonic()
    last_sent = None  # clock of the last frame sent (pts seconds for files, monotonic for live)
    try:
        while True:
            sample = appsink.try_pull_sample(Gst.SECOND // 4)
            if sample is None:
                msg = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
                if msg is not None and msg.type == Gst.MessageType.ERROR:
                    err, dbg = msg.parse_error()
                    print(f"gst_helper: error from {msg.src.get_name()}: {err.message} ({dbg})",
                          file=sys.stderr)
                    return 2
                if (msg is not None and msg.type == Gst.MessageType.EOS) or appsink.is_eos():
                    return 0
                if time.monotonic() - last_frame > args.timeout:
                    print(f"gst_helper: no frame for {args.timeout:.0f}s", file=sys.stderr)
                    return 3
                continue
            last_frame = time.monotonic()
            buf = sample.get_buffer()
            pts = buf.pts if buf.pts != Gst.CLOCK_TIME_NONE else -1

            if interval:
                clock = last_frame if args.live or pts < 0 else pts / 1e9
                if last_sent is not None and clock - last_sent < interval - 1e-3:
                    continue  # dropped before mapping: no copy, no conversion
                last_sent = clock

            info = GstVideo.VideoInfo.new_from_caps(sample.get_caps())
            meta = GstVideo.buffer_get_video_meta(buf)
            strides = list(meta.stride) if meta else list(info.stride)
            offsets = list(meta.offset) if meta else list(info.offset)
            w, h = info.width, info.height
            fps = info.fps_n / info.fps_d if info.fps_d else 0.0

            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                continue
            try:
                data = memoryview(mapinfo.data)
                if fmt == FORMATS["NV12"]:
                    y = data[offsets[0]: offsets[0] + strides[0] * h]
                    uv = data[offsets[1]: offsets[1] + strides[1] * (h // 2)]
                    planes, s1 = (y, uv), strides[1]
                else:
                    planes, s1 = (data[offsets[0]: offsets[0] + strides[0] * h],), 0
                size = sum(len(p) for p in planes)
                out.write(HEADER.pack(b"FRM2", fmt, w, h, strides[0], s1, size, pts, fps))
                for p in planes:
                    out.write(p)
            finally:
                buf.unmap(mapinfo)
    except (BrokenPipeError, KeyboardInterrupt):
        return 0
    finally:
        pipe.set_state(Gst.State.NULL)


if __name__ == "__main__":
    sys.exit(main())
