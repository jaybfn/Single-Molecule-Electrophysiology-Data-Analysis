"""Baseline estimation for ion-current traces."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class BaselineEstimator(Protocol):
    """Estimate a slowly varying open-pore baseline."""

    def estimate(self, current: NDArray[np.floating], sample_rate: float) -> NDArray[np.floating]:
        """Return baseline array with the same length as ``current``."""


class NoneBaseline:
    """No baseline correction — detector special-cases this to chunk mean."""

    def estimate(self, current: NDArray[np.floating], sample_rate: float) -> NDArray[np.floating]:
        return np.zeros_like(current, dtype=float)


class ConstantBaseline:
    """Use a single constant (default: median of the chunk/trace)."""

    def __init__(self, value: float | None = None):
        self.value = value

    def estimate(self, current: NDArray[np.floating], sample_rate: float) -> NDArray[np.floating]:
        level = float(np.median(current)) if self.value is None else float(self.value)
        return np.full_like(current, level, dtype=float)


class MedianBaseline:
    """Moving-median baseline for slow drift removal."""

    def __init__(self, window_s: float = 0.05):
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self.window_s = float(window_s)

    def estimate(self, current: NDArray[np.floating], sample_rate: float) -> NDArray[np.floating]:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        half = max(1, int(round(self.window_s * sample_rate / 2.0)))
        try:
            from scipy.ndimage import median_filter
        except ImportError as exc:  # pragma: no cover
            raise ImportError("scipy is required for MedianBaseline") from exc

        size = 2 * half + 1
        size = min(size, len(current) if len(current) % 2 == 1 else max(1, len(current) - 1))
        if size < 3:
            return np.full_like(current, float(np.median(current)), dtype=float)
        if size % 2 == 0:
            size += 1
        return median_filter(current.astype(float), size=size, mode="nearest").astype(float)


class PercentileBaseline:
    """Sliding-percentile open-pore estimate for long drift with frequent events.

    Events bias a plain mean/median toward the blocked level when occupancy is high.
    A high percentile (e.g. 90) tracks the open pore for downward events; a low
    percentile (e.g. 10) does the same for upward events.
    """

    def __init__(self, percentile: float = 90.0, window_s: float = 0.5):
        if not 0.0 <= percentile <= 100.0:
            raise ValueError("percentile must be in [0, 100]")
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self.percentile = float(percentile)
        self.window_s = float(window_s)

    def estimate(self, current: NDArray[np.floating], sample_rate: float) -> NDArray[np.floating]:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        x = np.asarray(current, dtype=float)
        n = len(x)
        if n == 0:
            return x
        win = max(3, int(round(self.window_s * sample_rate)))
        if win >= n:
            return np.full(n, float(np.percentile(x, self.percentile)), dtype=float)

        import pandas as pd

        s = pd.Series(x)
        bl = (
            s.rolling(window=win, center=True, min_periods=max(3, win // 5))
            .quantile(self.percentile / 100.0)
            .to_numpy(dtype=float)
        )
        if np.isnan(bl).any():
            global_p = float(np.percentile(x, self.percentile))
            bl = np.where(np.isnan(bl), global_p, bl)
        return bl


def residual_current(
    current: NDArray[np.floating],
    baseline: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return ``current - baseline``."""
    return np.asarray(current, dtype=float) - np.asarray(baseline, dtype=float)
