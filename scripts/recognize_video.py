"""Face recognition on a video file or a live RTSP stream.

    python scripts/recognize_video.py path/to/video.mp4 --out outputs/result.mp4
    python scripts/recognize_video.py rtsp://user:pass@192.168.1.10:554/stream1
    python scripts/recognize_video.py rtsp://... --events outputs/events.jsonl --show

Identity events (a track gets a name, or changes name) are printed and optionally
appended to a JSON-lines file. Ctrl+C stops cleanly.
"""

import argparse
import resource
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import cv2

from face_rec import FaceRecognizer, StreamRecognizer
from face_rec.config import load_config
from face_rec.sources import open_source
from face_rec.utils.draw import draw_tracks

log = logging.getLogger("recognize_video")


def cpu_seconds(src=None) -> float:
    """CPU time of this process, plus the GStreamer helper process if there is one."""
    r = resource.getrusage(resource.RUSAGE_SELF)
    total = r.ru_utime + r.ru_stime
    proc = getattr(src, "proc", None)
    if proc is not None:
        try:
            fields = open(f"/proc/{proc.pid}/stat").read().rsplit(")", 1)[1].split()
            total += (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
        except (OSError, IndexError, ValueError):
            pass
    return total


def compact_timing(rows: dict) -> str:
    """'name mean/p95' for the main rows, e.g. 'frame 9.1/14.2 detect 7.0/11.3 (npu 3.4/3.6)'."""
    parts = []
    for name, r in rows.items():
        label = name.strip().split()[1] if "infer" in name else name
        text = f"{r['mean']:.1f}/{r['p95']:.1f}"
        if "infer" in name:
            parts[-1] += f" (npu {text})"
        elif name in ("frame", "latency", "detect", "embed"):
            parts.append(f"{label} {text}")
    return "ms mean/p95: " + " ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="video file, rtsp:// URL, or camera index")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", help="write annotated video (.mp4)")
    p.add_argument("--events", help="append identity events to this .jsonl file")
    p.add_argument("--show", action="store_true", help="display a window (needs a display)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames read")
    p.add_argument("--duration", type=float, default=0, help="stop after N seconds")
    p.add_argument("--stats-every", type=float, default=5.0, help="seconds between stat lines")
    p.add_argument("--timing-log", help="write per-inferred-frame timings to this .csv file")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.show and not os.environ.get("DISPLAY"):
        log.warning("--show ignored: no DISPLAY")
        args.show = False

    cfg = load_config(args.config)
    fr = FaceRecognizer(cfg)
    if not fr.db.names():
        log.warning("Database is empty: every face will be 'unknown'. Run scripts/enroll.py first.")
    sr = StreamRecognizer(fr, cfg.get("tracking"))

    writer = None
    events_file = None
    if args.events:
        Path(args.events).parent.mkdir(parents=True, exist_ok=True)
        events_file = open(args.events, "a", encoding="utf-8")

    timing_file = None
    if args.timing_log:
        Path(args.timing_log).parent.mkdir(parents=True, exist_ok=True)
        timing_file = open(args.timing_log, "w", encoding="utf-8")
        timing_file.write("frame_index,timestamp,faces,embedded,total_ms,detect_ms,"
                          "detect_infer_ms,embed_ms,embed_infer_ms,latency_ms\n")

    processed, t_start = 0, time.time()
    t_stats, n_stats = t_start, 0
    det_stats = emb_stats = 0
    ts_stats = None
    fps_now = 0.0
    try:
        with open_source(args.source, cfg.get("source")) as src:
            reconnects = src.reconnects
            cpu_start, wall_start = cpu_seconds(src), time.time()
            for frame in src:
                if src.reconnects != reconnects:  # stream was reopened: old tracks are stale
                    reconnects = src.reconnects
                    sr.reset()

                if ts_stats is None:
                    ts_stats = frame.timestamp
                result = sr.process(frame.image, frame.timestamp)
                processed += 1
                n_stats += 1
                if result.inferred:
                    # live: time from the frame being received to its result being ready
                    latency = time.time() - frame.timestamp if src.is_live else None
                    if latency is not None:
                        fr.timer.add("latency", latency)
                    if timing_file:
                        t = result.timing
                        timing_file.write(
                            f"{frame.index},{frame.timestamp:.3f},{t['faces']},{t['embedded']},"
                            f"{t['total']:.2f},{t['detect']:.2f},{t['detect_infer']:.2f},"
                            f"{t['embed']:.2f},{t['embed_infer']:.2f},"
                            f"{'' if latency is None else f'{1000 * latency:.2f}'}\n")

                for ev in result.events:
                    log.info("track #%d -> %s (%.2f) at frame %d", ev.track_id, ev.name,
                             ev.similarity, frame.index)
                    if events_file:
                        events_file.write(json.dumps({"source": src.description, **asdict(ev)}) + "\n")
                        events_file.flush()

                if args.out or args.show:
                    fps_text = f"{fps_now:.1f} FPS" if fps_now else "-- FPS"
                    info = f"{fps_text}  faces: {len(result.tracks)}"
                    vis = draw_tracks(frame.image, result.tracks, info)
                    if args.out:
                        if writer is None:
                            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                            h, w = vis.shape[:2]
                            # gstreamer max_fps thins the stream: write at the delivered rate
                            limit = float(((cfg.get("source") or {}).get("gstreamer") or {})
                                          .get("max_fps", 0) or 0)
                            fps_out = min(src.fps, limit) if limit and getattr(src, "proc", None) else src.fps
                            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                                     fps_out, (w, h))
                        writer.write(vis)
                    if args.show:
                        cv2.imshow("face_rec", vis)
                        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            break

                now = time.time()
                if now - t_stats >= args.stats_every:
                    dt = now - t_stats
                    fps_now = n_stats / dt
                    # NPU rates per second of video time (== wall time for live streams)
                    media_dt = max(frame.timestamp - ts_stats, 1e-6)
                    det_rate = (sr.detect_calls - det_stats) / media_dt
                    emb_rate = (sr.embed_calls - emb_stats) / media_dt
                    timing = compact_timing(fr.timing_stats())
                    progress = (f"frame {frame.index + 1}/{src.frame_count}" if src.frame_count
                                else f"dropped={src.dropped} reconnects={src.reconnects}")
                    log.info("%.1f FPS read | NPU detect %.1f/s embed %.1f/s (per video s)%s | %s | %s",
                             fps_now, det_rate, emb_rate, " (idle)" if sr.idle else "",
                             progress, timing)
                    t_stats, n_stats, ts_stats = now, 0, frame.timestamp
                    det_stats, emb_stats = sr.detect_calls, sr.embed_calls

                if args.max_frames and processed >= args.max_frames:
                    break
                if args.duration and time.time() - wall_start >= args.duration:
                    break
            cpu_used = cpu_seconds(src) - cpu_start
            log.info("CPU: %.0f%% of one core on average (%s source)",
                     100 * cpu_used / max(time.time() - wall_start, 1e-6),
                     (cfg.get("source") or {}).get("backend", "opencv"))
    except KeyboardInterrupt:
        log.info("Interrupted")
    except ConnectionError as e:
        log.error("%s", e)
    finally:
        elapsed = time.time() - t_start
        log.info("Read %d frames in %.1fs; NPU calls: detect %d, embed %d", processed, elapsed,
                 sr.detect_calls, sr.embed_calls)
        if sr.detect_calls:
            print("\nTiming (inferred frames only; percentiles over the last 2000 calls)\n"
                  + fr.timing_report())
        if writer is not None:
            writer.release()
            log.info("Saved %s", args.out)
        if timing_file:
            timing_file.close()
            log.info("Saved %s", args.timing_log)
        if events_file:
            events_file.close()
        if args.show:
            cv2.destroyAllWindows()
        fr.close()


if __name__ == "__main__":
    main()
