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
    """No baseline correction — returns zeros (identity residual is the signal itself).

    When used by :class:`~pynanopore.detection.events.EventDetector`, a ``NoneBaseline``
    causes detection to run on the raw current (baseline treated as constant 0 in residual
    space is not used; the detector special-cases this).
    """

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
    """Moving-median baseline for slow drift removal.

    Parameters
    ----------
    window_s:
        Median filter window in seconds. Larger windows follow slower drift only.
    """

    def __init__(self, window_s: float = 0.05):
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self.window_s = float(window_s)

    def estimate(self, current: NDArray[np.floating], sample_rate: float) -> NDArray[np.floating]:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        half = max(1, int(round(self.window_s * sample_rate / 2.0)))
        # Reflect-pad then sliding median via scipy if available; else numpy stride
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


def residual_current(
    current: NDArray[np.floating],
    baseline: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return ``current - baseline``."""
    return np.asarray(current, dtype=float) - np.asarray(baseline, dtype=float)
