# face_rec

Modular face recognition for the Qualcomm RB5 Gen2 (QCS8550). The same pipeline runs
on CPU with ONNX Runtime and on the HTP NPU with TFLite + QNN or QNN binaries.

```
image → Detector → Alignment (5 pts → 112×112) → Recognizer → FaceDatabase → name + similarity
```

Each stage is pluggable and selected in `configs/default.yaml`:

| Axis | Registry | Built-in |
|---|---|---|
| Inference engine | `face_rec.engines.ENGINES` | `onnxruntime_qnn` (HTP NPU), `onnxruntime` (CPU), `tflite_qnn`*, `qnn`* |
| Detector | `face_rec.detection.DETECTORS` | `scrfd` |
| Recognizer | `face_rec.recognition.RECOGNIZERS` | `arcface`, `cavaface`* |
| Database | `face_rec.database.DATABASES` | `sqlite` |

\* not implemented yet (see roadmap).

## Layout

```
configs/           YAML configs
models/            model files (not committed): detection/, recognition/
data/gallery/      enrollment images: <person>/*.jpg
data/db/           embedding database
src/face_rec/
  engines/         InferenceEngine + registry + fallback builder
  detection/       Detector interface, SCRFD
  alignment/       5-point similarity transform, templates
  recognition/     Recognizer (ArcFace-style default), CavaFace
  database/        FaceDatabase interface, SQLite backend
  sources/         open_source(): OpenCV or GStreamer (Qualcomm HW decode) backends,
                   files + RTSP, reconnect, newest-frame; gst_helper.py runs in system Python
  tracking/        Tracker interface, IoU tracker, Track (name voting)
  stream.py        StreamRecognizer: detect -> track -> embed when due -> events
  utils/           drawing, timing
  pipeline.py      FaceRecognizer
scripts/           download_models, enroll, recognize_*, benchmark, export/
tests/
```

## Setup

```bash
conda create -n face python=3.11 -y && conda activate face
pip install -r requirements.txt
pip install -e .
pytest
```

## Usage

```bash
# 1. enroll people (images are copied to data/gallery/<name>/)
python scripts/enroll.py --name ntrthanh --dir assets/ntrthanh_face
python scripts/enroll.py --list

# 2. video file
python scripts/recognize_video.py input.mp4 --out outputs/result.mp4 --events outputs/events.jsonl

# 3. RTSP stream (reconnects automatically; Ctrl+C to stop)
python scripts/recognize_video.py "rtsp://user:pass@192.168.1.10:554/stream1" --events outputs/events.jsonl
```

Each frame is detected. Faces are tracked by IoU and embedded only when a track appears,
then every `reembed_interval` frames (or `unknown_reembed_interval` while unknown). A track's
name is the majority vote of its last `vote_window` matches. An event is emitted when a track
gets a name or changes name. Live sources always process the newest frame, so if processing
is slower than the stream, stale frames are dropped instead of building up latency.

## Video input: GStreamer + Qualcomm hardware decode

`source.backend: gstreamer` (the default) decodes with `qtic2vdec`, the Qualcomm hardware
decoder. The pipeline runs in a helper process (`sources/gst_helper.py`) under the system
Python, because only that Python has the vendor GStreamer bindings. The helper drops frames
down to `gstreamer.max_fps` *before* copying them, and sends NV12 frames, which are converted
to BGR with OpenCV only for the frames that are used.

Measured on RB5 Gen2 with a 1080p RTSP stream at 30 fps, full recognition pipeline:

| input | CPU (one core = 100%) |
|---|---|
| OpenCV/FFmpeg, OpenCV 6 threads (before) | 99% |
| OpenCV/FFmpeg, `opencv_threads: 1` | 74% |
| GStreamer qtic2vdec NV12, all frames | 70% |
| **GStreamer qtic2vdec NV12, `max_fps: 5`** | **31%** |

Notes:
- `qtivtransform` with `engine=gles` hangs on this box, `engine=fcv` works but shifts
  colors, and `videorate` after `qtic2vdec` crashes (buffers have no duration). That is why
  frames are dropped in the helper and converted in OpenCV.
- Hardware decoders pad strides (for example, 480-wide NV12 has stride 512). The helper uses
  each buffer's VideoMeta.
- `opencv_threads: 1` matters: OpenCV's worker threads busy-wait between calls. With sparse
  calls, that burns about 20x the CPU of the work itself.
- With less CPU spinning, the CPU clocks down between frames, so per-frame latency goes from
  about 15 ms to about 30 ms (p50). That is well within the 200 ms between detections.
- Set `max_fps: 0` to get every frame, for example for a smooth `--out` video. Use
  `backend: opencv` on machines without the Qualcomm stack.

## NPU notes (onnxruntime-qnn)

- Python >= 3.11 (required by `onnxruntime-qnn`).
- HTP needs static shapes: set `input_shapes` in the engine options.
- Do not set `ADSP_LIBRARY_PATH`; the EP points it at the QNN libs bundled in the wheel.
- `allow_cpu_fallback: false` (the default) makes loading fail if any node can't run on QNN,
  so the config falls through to the CPU engine instead of silently splitting the graph.
- Float models run in FP16 on HTP. INT8 (QDQ) models are faster still (phase 3).

## Extending

**New engine**: subclass `InferenceEngine` in `engines/xxx_engine.py`. Implement `is_available`,
`load`, `inputs`, `outputs` and `_run_raw`, and decorate the class with `@ENGINES.register("xxx")`.
Then import the module in `engines/__init__.py`. The base class converts NCHW/NHWC layouts and
handles INT8 quantization.

**New database**: subclass `FaceDatabase` in `database/xxx_db.py`, register it with
`@DATABASES.register("xxx")`, import it in `database/__init__.py`, and set `database.type`.

**New model**: subclass `Detector` or `Recognizer`, register it, and set `type:` in the config.
If you change the recognizer, run `scripts/enroll.py --rebuild`, because embeddings from
different models can't be compared. The database stores `model_id` and refuses to load when it
doesn't match.

## Roadmap

- [x] Phase 0: environment, download buffalo_s (det_500m + w600k_mbf)
- [x] Phase 1: SCRFD, SQLite DB, enroll, recognize_image
- [x] Phase 2: video file + RTSP sources, IoU tracking, identity events
- [ ] Threshold tuning on real data (scripts/evaluate.py), hardware video decode
- [x] onnxruntime-qnn engine on HTP, benchmark
- [ ] Phase 3: INT8 export, EP context cache
- [ ] Phase 4: CavaFace, API, anti-spoofing, FAISS DB
