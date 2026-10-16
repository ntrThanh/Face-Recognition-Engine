"""Benchmark every engine listed in the config: load time, latency, and accuracy vs CPU.

    python scripts/benchmark.py                      # detector + recognizer
    python scripts/benchmark.py --only recognizer --runs 100
"""

import argparse
import time

import numpy as np

from face_rec.config import load_config, resolve_path
from face_rec.engines import ENGINES

REFERENCE = "onnxruntime"


def build(candidate):
    cls = ENGINES.get(candidate["name"])
    if not cls.is_available():
        raise RuntimeError("not available on this machine")
    engine = cls()
    t = time.perf_counter()
    engine.load(str(resolve_path(candidate["path"])), candidate.get("options") or {})
    return engine, time.perf_counter() - t


def input_shape(engine):
    shape = engine.inputs()[0].shape
    if not all(isinstance(d, int) and d > 0 for d in shape):
        raise ValueError(f"dynamic input shape {shape}: set `input_shapes` in options")
    if engine.inputs()[0].layout == "NHWC":
        shape = (shape[0], shape[3], shape[1], shape[2])
    return shape


def latency_ms(engine, x, runs):
    engine.warmup()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        engine.run(x)
        times.append(time.perf_counter() - t)
    times = np.array(times) * 1000
    return float(np.mean(times)), float(np.percentile(times, 95))


def compare(outs, ref):
    max_diff = max(float(np.abs(o - r).max()) for o, r in zip(outs, ref))
    a, b = np.concatenate([o.ravel() for o in outs]), np.concatenate([r.ravel() for r in ref])
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return max_diff, cosine


def bench_component(key, cfg, runs):
    print(f"\n== {key} ({cfg['type']}) ==")
    rng = np.random.default_rng(0)
    results, x, ref = [], None, None
    # Run the CPU reference first so every other engine can be compared to it.
    candidates = sorted(cfg["engines"], key=lambda c: c["name"] != REFERENCE)
    for c in candidates:
        try:
            engine, load_s = build(c)
            if x is None:
                x = rng.uniform(-1, 1, input_shape(engine)).astype(np.float32)
            outs = engine.run(x)
            mean, p95 = latency_ms(engine, x, runs)
            providers = engine.providers() if hasattr(engine, "providers") else []
            if c["name"] == REFERENCE:
                ref = outs
            acc = "reference" if c["name"] == REFERENCE else (
                "max|diff|={:.4f} cosine={:.4f}".format(*compare(outs, ref)) if ref else "-"
            )
            results.append((c["name"], f"{load_s:.2f}s", f"{mean:.2f}", f"{p95:.2f}",
                            f"{1000 / mean:.0f}", acc, providers[0] if providers else ""))
            engine.close()
        except Exception as e:
            results.append((c["name"], "FAIL", "-", "-", "-", str(e).splitlines()[0][:80], ""))

    header = ("engine", "load", "mean ms", "p95 ms", "FPS", "accuracy vs CPU", "provider")
    widths = [max(len(str(r[i])) for r in [header, *results]) for i in range(len(header))]
    for row in [header, *results]:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(row, widths)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--only", choices=["detector", "recognizer"])
    p.add_argument("--runs", type=int, default=50)
    args = p.parse_args()

    cfg = load_config(args.config)
    for key in ["detector", "recognizer"]:
        if args.only in (None, key):
            bench_component(key, cfg[key], args.runs)


if __name__ == "__main__":
    main()
