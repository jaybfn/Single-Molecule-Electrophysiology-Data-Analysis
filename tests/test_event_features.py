"""Tests for baseline, rich event features, and pulse-shape idealization."""

from __future__ import annotations

import numpy as np
import pytest

from pynanopore.detection.baseline import ConstantBaseline, MedianBaseline, NoneBaseline
from pynanopore.detection.events import EventDetector
from pynanopore.detection.pulse_shape import PulseShapeIdealizer
from pynanopore.io.trace import Trace


def _make_trace(
    *,
    fs: float = 1000.0,
    duration_s: float = 2.0,
    open_level: float = 100.0,
    direction: str = "down",
    depth: float = 40.0,
    events: list[tuple[float, float]] | None = None,
    drift: float = 0.0,
    noise: float = 1.0,
    seed: int = 0,
) -> Trace:
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n, dtype=float) / fs
    baseline = open_level + drift * t
    current = baseline + rng.normal(0, noise, size=n)
    if events is None:
        events = [(0.4, 0.05), (1.0, 0.08)]
    for start_s, width_s in events:
        a = int(start_s * fs)
        b = int((start_s + width_s) * fs)
        if direction == "down":
            current[a:b] -= depth
        else:
            current[a:b] += depth
    return Trace(time=t, current=current, sample_rate=fs, source="synthetic")


def test_rich_features_downward():
    trace = _make_trace(direction="down", open_level=100.0, depth=40.0)
    det = EventDetector(0.5, 2.0, min_duration=0.02, direction="down")
    events = det.detect_trace(trace, interval_length=2.0)
    assert len(events) >= 1
    ev = events[0]
    assert ev.dwell_time == pytest.approx(ev.difference)
    assert ev.delta_i > 10
    assert ev.area > 0
    assert ev.i0 == pytest.approx(100.0, abs=5.0)
    assert ev.blockade_mean < ev.i0
    assert "delta_i_over_i0" in ev.to_dict()


def test_upward_pulses_like_screenshot():
    # Negative open pore, pulses toward less-negative values
    trace = _make_trace(
        direction="up",
        open_level=-565.0,
        depth=140.0,
        noise=2.0,
        events=[(0.3, 0.04), (0.7, 0.06), (1.2, 0.05)],
    )
    det = EventDetector(
        0.4,
        1.5,
        min_duration=0.02,
        direction="up",
        baseline=ConstantBaseline(-565.0),
    )
    events = det.detect_trace(trace, interval_length=2.0)
    assert len(events) >= 2
    assert all(e.blockade_mean > e.i0 for e in events)


def test_median_baseline_with_drift():
    trace = _make_trace(direction="down", drift=20.0, depth=50.0, noise=0.5)
    det = EventDetector(
        0.5,
        2.0,
        min_duration=0.02,
        direction="down",
        baseline=MedianBaseline(window_s=0.1),
    )
    events = det.detect_trace(trace, interval_length=2.0)
    assert len(events) >= 1


def test_pulse_shape_levels():
    trace = _make_trace(direction="down", open_level=100.0, depth=40.0, noise=0.2)
    det = EventDetector(0.5, 2.0, min_duration=0.02, direction="down")
    events = det.detect_trace(trace, interval_length=2.0)
    assert events
    pulse = PulseShapeIdealizer.from_events(trace, events)
    assert pulse.idealized.shape == trace.current.shape
    # Inside first event, idealized ≈ blockade_mean
    ev = events[0]
    mid = (ev.start_idx + ev.end_idx) // 2
    assert pulse.idealized[mid] == pytest.approx(ev.blockade_mean)
    # Outside events near start, open level
    assert pulse.idealized[0] == pytest.approx(np.median([e.i0 for e in events]), abs=1e-6)
    rt, _ = pulse.rising_edges
    ft, _ = pulse.falling_edges
    assert len(rt) == len(events)
    assert len(ft) == len(events)


def test_none_baseline_compatible():
    trace = _make_trace()
    det = EventDetector(0.5, 2.0, min_duration=0.02, baseline=NoneBaseline())
    events = det.detect_trace(trace, interval_length=2.0)
    assert isinstance(events, list)


def test_overlap_dedup():
    trace = _make_trace(events=[(0.9, 0.05)])
    det = EventDetector(0.5, 2.0, min_duration=0.02)
    events = det.detect_trace(trace, interval_length=1.0, overlap=0.2)
    starts = [e.start_idx for e in events]
    assert len(starts) == len(set(starts))
