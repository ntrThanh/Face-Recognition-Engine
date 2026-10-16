"""Lightweight per-stage timing with percentiles over a recent window."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import contextmanager

import numpy as np


class StageTimer:
    def __init__(self, window: int = 2000) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.samples: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    @contextmanager
    def __call__(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, time.perf_counter() - t0)

    def add(self, stage: str, seconds: float) -> None:
        self.totals[stage] += seconds
        self.counts[stage] += 1
        self.samples[stage].append(seconds)

    def summary(self) -> dict[str, float]:
        """Average milliseconds per stage."""
        return {k: 1000 * self.totals[k] / self.counts[k] for k in self.totals}

    def stats(self) -> dict[str, dict[str, float]]:
        """Per stage: calls, mean, p50, p95, max (ms; percentiles over the recent window)."""
        out = {}
        for k, n in self.counts.items():
            s = np.asarray(self.samples[k]) * 1000
            out[k] = {"calls": n, "mean": 1000 * self.totals[k] / n,
                      "p50": float(np.percentile(s, 50)), "p95": float(np.percentile(s, 95)),
                      "max": float(s.max())}
        return out

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()
        self.samples.clear()


def format_stats(rows: dict[str, dict[str, float]]) -> str:
    """Render StageTimer.stats()-style rows as a fixed-width table."""
    header = f"{'stage':<26}{'calls':>7}{'mean':>9}{'p50':>9}{'p95':>9}{'max':>9}  (ms)"
    lines = [header, "-" * len(header)]
    for name, r in rows.items():
        lines.append(f"{name:<26}{r['calls']:>7}{r['mean']:>9.2f}{r['p50']:>9.2f}"
                     f"{r['p95']:>9.2f}{r['max']:>9.2f}")
    return "\n".join(lines)
