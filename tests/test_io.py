"""Tests for Trace I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pynanopore.io.readers import load_trace
from pynanopore.io.trace import Trace


def test_trace_validation():
    with pytest.raises(ValueError):
        Trace(time=np.array([1.0]), current=np.array([1.0, 2.0]), sample_rate=1000)


def test_load_csv(csv_trace_path: Path):
    trace = load_trace(csv_trace_path)
    assert len(trace.time) > 0
    assert trace.sample_rate > 0
    assert trace.time.shape == trace.current.shape


def test_load_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_trace(tmp_path / "missing.csv")


def test_unsupported_extension(tmp_path: Path):
    bad = tmp_path / "x.txt"
    bad.write_text("nope")
    with pytest.raises(ValueError, match="Unsupported"):
        load_trace(bad)
