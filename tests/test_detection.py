"""Tests for event detection."""

from __future__ import annotations

from pynanopore.detection.chunking import ChunkGenerator
from pynanopore.detection.events import EventDetector
from pynanopore.io.trace import Trace


def test_detect_synthetic_events(synthetic_trace: Trace):
    detector = EventDetector(std_multiplier=0.5, threshold_multiplier=2.0, min_duration=0.01)
    events = detector.detect_trace(synthetic_trace, interval_length=1.0)
    assert len(events) >= 1
    assert all(e.difference >= 0.01 for e in events)
    assert all(e.end_time > e.start_time for e in events)


def test_detect_empty_chunk():
    detector = EventDetector()
    assert detector.detect_events([], []) == []  # type: ignore[arg-type]


def test_chunk_generator(synthetic_trace: Trace):
    chunker = ChunkGenerator(synthetic_trace.sample_rate, interval_length=0.5)
    chunks = list(chunker.generate_from_trace(synthetic_trace))
    assert len(chunks) >= 1
    assert sum(len(c.current) for c in chunks) == len(synthetic_trace.current)


def test_detect_events_dicts(synthetic_trace: Trace):
    detector = EventDetector(std_multiplier=0.5, threshold_multiplier=2.0)
    chunk = synthetic_trace.current[:500]
    time = synthetic_trace.time[:500]
    dicts = detector.detect_events_dicts(chunk, time)
    assert isinstance(dicts, list)
    if dicts:
        assert "difference" in dicts[0]
