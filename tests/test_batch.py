"""Tests for batch detection pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pynanopore.batch import BatchDetectConfig, batch_detect, discover_recordings


def _write_csv(path: Path, *, open_level: float = 100.0, depth: float = 40.0) -> None:
    fs = 1000.0
    n = 2000
    t = np.arange(n) / fs
    current = np.full(n, open_level, dtype=float)
    current[400:450] -= depth
    current[1200:1280] -= depth
    pd.DataFrame({"time_column": t, "data_column": current}).to_csv(path, index=False)


def test_batch_detect(tmp_path: Path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    _write_csv(in_dir / "a.csv")
    _write_csv(in_dir / "b.csv")

    files = discover_recordings(in_dir)
    assert len(files) == 2

    summary = batch_detect(
        in_dir,
        out_dir,
        BatchDetectConfig(direction="down", interval_length=2.0, fit_dwelltime=True),
    )
    assert len(summary) == 2
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "run_metadata.json").exists()
    assert (out_dir / "events" / "a_events.csv").exists()
    assert int(summary["n_events"].sum()) >= 1
