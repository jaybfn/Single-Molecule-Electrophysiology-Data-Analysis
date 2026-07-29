"""Shared data models for ion-current traces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Trace:
    """A continuous ion-current recording."""

    time: NDArray[np.floating]
    current: NDArray[np.floating]
    sample_rate: float
    source: str = ""

    def __post_init__(self) -> None:
        if self.time.shape != self.current.shape:
            raise ValueError("time and current arrays must have the same shape")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if len(self.time) == 0:
            raise ValueError("trace must contain at least one sample")

    @property
    def duration(self) -> float:
        """Recording duration in seconds."""
        return float(self.time[-1] - self.time[0]) if len(self.time) > 1 else 0.0

    def slice_by_index(self, start: int, end: int) -> Trace:
        """Return a sub-trace by sample indices."""
        return Trace(
            time=self.time[start:end],
            current=self.current[start:end],
            sample_rate=self.sample_rate,
            source=self.source,
        )
