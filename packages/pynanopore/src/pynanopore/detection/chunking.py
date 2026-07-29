"""Chunk continuous traces into fixed-duration windows."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from pynanopore.io.trace import Trace


class ChunkGenerator:
    """Yield successive windows of a Trace for manageable analysis."""

    def __init__(self, sample_rate: float, interval_length: float = 5.0):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if interval_length <= 0:
            raise ValueError("interval_length must be positive")
        self.sample_rate = float(sample_rate)
        self.interval_length = float(interval_length)
        self.points_per_interval = max(1, int(self.sample_rate * self.interval_length))

    def generate(
        self, current: NDArray[np.floating], time: NDArray[np.floating]
    ) -> Iterator[tuple[NDArray[np.floating], NDArray[np.floating]]]:
        """Yield ``(current_chunk, time_chunk)`` pairs."""
        step = self.points_per_interval
        for start in range(0, len(current), step):
            end = start + step
            yield current[start:end], time[start:end]

    def generate_from_trace(self, trace: Trace) -> Iterator[Trace]:
        """Yield Trace slices covering ``trace``."""
        step = self.points_per_interval
        for start in range(0, len(trace.current), step):
            end = min(start + step, len(trace.current))
            yield trace.slice_by_index(start, end)


# Backward-compatible alias
class CreatingChunks(ChunkGenerator):
    """Deprecated: prefer :class:`ChunkGenerator`."""

    def __init__(self, abf, interval_length: float = 5.0):
        sample_rate = float(abf.dataRate) if hasattr(abf, "dataRate") else float(abf)
        super().__init__(sample_rate=sample_rate, interval_length=interval_length)
        self.abf = abf

    def generate_chunks(self, sweep_data: NDArray[np.floating]):
        step = self.points_per_interval
        for start in range(0, len(sweep_data), step):
            yield sweep_data[start : start + step]
