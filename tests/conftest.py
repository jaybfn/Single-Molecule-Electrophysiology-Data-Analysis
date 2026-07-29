"""Synthetic fixtures for unit tests (no large ABF binaries)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pynanopore.io.trace import Trace


@pytest.fixture
def sample_rate() -> float:
    return 1000.0


@pytest.fixture
def synthetic_trace(sample_rate: float) -> Trace:
    """Baseline current with a few clear downward events."""
    duration_s = 2.0
    n = int(duration_s * sample_rate)
    t = np.arange(n, dtype=float) / sample_rate
    current = np.full(n, 100.0, dtype=float)
    rng = np.random.default_rng(42)
    current += rng.normal(0, 1.0, size=n)

    # Inject three rectangular blockade events
    for start_s, width_s, depth in [(0.3, 0.05, 40.0), (0.8, 0.08, 50.0), (1.4, 0.04, 35.0)]:
        start = int(start_s * sample_rate)
        end = int((start_s + width_s) * sample_rate)
        current[start:end] -= depth

    return Trace(time=t, current=current, sample_rate=sample_rate, source="synthetic")


@pytest.fixture
def events_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"difference": rng.exponential(scale=0.01, size=500)})


@pytest.fixture
def csv_trace_path(tmp_path: Path, synthetic_trace: Trace) -> Path:
    path = tmp_path / "trace.csv"
    pd.DataFrame(
        {
            "time_column": synthetic_trace.time,
            "data_column": synthetic_trace.current,
        }
    ).to_csv(path, index=False)
    return path
