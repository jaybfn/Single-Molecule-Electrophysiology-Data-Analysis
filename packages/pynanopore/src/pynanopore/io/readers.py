"""Load electrophysiology recordings into a common Trace model."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyabf

from pynanopore.io.trace import Trace


def load_trace(
    file_path: str | Path,
    *,
    sweep: int = 0,
    invert_negative: bool = True,
    time_column: str = "time_column",
    data_column: str = "data_column",
    sample_rate: float | None = None,
) -> Trace:
    """
    Load an ABF or CSV recording as a Trace.

    Parameters
    ----------
    file_path:
        Path to ``.abf`` or ``.csv`` file.
    sweep:
        ABF sweep index to load (ignored for CSV).
    invert_negative:
        If True and the mean current is negative, multiply by -1 so
        events appear as downward (or consistent) polarity.
    time_column / data_column:
        Column names for CSV files.
    sample_rate:
        Required for CSV if not inferable from time spacing.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext == ".abf":
        return _load_abf(path, sweep=sweep, invert_negative=invert_negative)
    if ext == ".csv":
        return _load_csv(
            path,
            time_column=time_column,
            data_column=data_column,
            sample_rate=sample_rate,
            invert_negative=invert_negative,
        )
    raise ValueError(f"Unsupported file format: {ext}")


def _load_abf(path: Path, *, sweep: int, invert_negative: bool) -> Trace:
    abf = pyabf.ABF(str(path))
    if sweep not in abf.sweepList:
        raise ValueError(f"Sweep {sweep} not in ABF sweepList {list(abf.sweepList)}")
    abf.setSweep(sweep)
    current = np.asarray(abf.sweepY, dtype=float)
    time = np.asarray(abf.sweepX, dtype=float)
    if invert_negative and float(np.mean(current)) < 0:
        current = np.asarray(-current, dtype=float)
    return Trace(
        time=time,
        current=current,
        sample_rate=float(abf.dataRate),
        source=str(path),
    )


def _load_csv(
    path: Path,
    *,
    time_column: str,
    data_column: str,
    sample_rate: float | None,
    invert_negative: bool,
) -> Trace:
    df = pd.read_csv(path)
    if time_column not in df.columns or data_column not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{time_column}' and '{data_column}'. "
            f"Found: {list(df.columns)}"
        )
    time = np.asarray(df[time_column].to_numpy(dtype=float), dtype=float)
    current = np.asarray(df[data_column].to_numpy(dtype=float), dtype=float)
    if invert_negative and float(np.mean(current)) < 0:
        current = np.asarray(-current, dtype=float)

    if sample_rate is None:
        if len(time) < 2:
            raise ValueError("Cannot infer sample_rate from a single-point CSV; pass sample_rate=")
        diffs = np.diff(time)
        median_dt = float(np.median(diffs))
        if median_dt <= 0:
            raise ValueError("Cannot infer sample_rate from non-increasing time column")
        sample_rate = 1.0 / median_dt

    return Trace(time=time, current=current, sample_rate=float(sample_rate), source=str(path))


# Backward-compatible alias used by older scripts
class ReadingData:
    """Deprecated wrapper around :func:`load_trace` for ABF/CSV paths."""

    def __init__(self, file_path: str | os.PathLike[str]):
        self._path = Path(file_path)
        ext = self._path.suffix.lower()
        if ext == ".abf":
            self.data = pyabf.ABF(str(self._path))
        elif ext == ".csv":
            self.data = pd.read_csv(self._path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def get_data(self):
        return self.data

    def to_trace(self, sweep: int = 0, invert_negative: bool = True) -> Trace:
        return load_trace(self._path, sweep=sweep, invert_negative=invert_negative)
