"""Threshold-based event detection on ion-current chunks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from pynanopore.detection.chunking import ChunkGenerator
from pynanopore.io.trace import Trace


@dataclass
class Event:
    """A detected translocation / blockade event."""

    start_time: float
    end_time: float
    difference: float
    amplitude: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class EventDetector:
    """
    Detect events where current drops below a std-based entry threshold
    and reaches a deeper absolute threshold before recovering.
    """

    def __init__(
        self,
        std_multiplier: float = 0.25,
        threshold_multiplier: float = 1.5,
        min_duration: float = 1e-4,
    ):
        if std_multiplier < 0 or threshold_multiplier < 0:
            raise ValueError("multipliers must be non-negative")
        if min_duration < 0:
            raise ValueError("min_duration must be non-negative")
        self.std_multiplier = float(std_multiplier)
        self.threshold_multiplier = float(threshold_multiplier)
        self.min_duration = float(min_duration)

    def detect_events(
        self,
        data_chunk: NDArray[np.floating],
        data_time: NDArray[np.floating],
    ) -> list[Event]:
        """Detect events in a single chunk; returns Event objects."""
        if len(data_chunk) < 2:
            return []
        if len(data_chunk) != len(data_time):
            raise ValueError("data_chunk and data_time must have the same length")

        mean = float(np.mean(data_chunk))
        std_dev = float(np.std(data_chunk))
        threshold = mean - self.threshold_multiplier * std_dev
        std_threshold = mean - self.std_multiplier * std_dev

        events: list[Event] = []
        start_time: float | None = None
        start_idx: int | None = None
        crossed_threshold = False

        for i in range(1, len(data_chunk)):
            if data_chunk[i] < std_threshold and data_chunk[i - 1] >= std_threshold:
                start_time = float(data_time[i])
                start_idx = i
                crossed_threshold = False

            if start_time is not None and data_chunk[i] < threshold:
                crossed_threshold = True

            if (
                start_time is not None
                and start_idx is not None
                and data_chunk[i] >= std_threshold
                and data_chunk[i - 1] < std_threshold
            ):
                if crossed_threshold:
                    end_time = float(data_time[i])
                    duration = end_time - start_time
                    if duration >= self.min_duration:
                        segment = data_chunk[start_idx : i + 1]
                        events.append(
                            Event(
                                start_time=start_time,
                                end_time=end_time,
                                difference=duration,
                                amplitude=float(np.min(segment)),
                            )
                        )
                start_time = None
                start_idx = None
                crossed_threshold = False

        return events

    def detect_events_dicts(
        self,
        data_chunk: NDArray[np.floating],
        data_time: NDArray[np.floating],
    ) -> list[dict[str, float]]:
        """Backward-compatible list-of-dicts API."""
        return [e.to_dict() for e in self.detect_events(data_chunk, data_time)]

    def detect_trace(
        self,
        trace: Trace,
        *,
        interval_length: float = 5.0,
    ) -> list[Event]:
        """Run detection over an entire Trace using fixed-length chunks."""
        chunker = ChunkGenerator(trace.sample_rate, interval_length=interval_length)
        all_events: list[Event] = []
        for current_chunk, time_chunk in chunker.generate(trace.current, trace.time):
            all_events.extend(self.detect_events(current_chunk, time_chunk))
        return all_events


# Backward-compatible alias
class EventDetection(EventDetector):
    """Deprecated: prefer :class:`EventDetector`."""

    def detect_events(  # type: ignore[override]
        self,
        data_chunk: NDArray[np.floating],
        data_time: NDArray[np.floating],
    ) -> list[dict[str, float]]:
        return self.detect_events_dicts(data_chunk, data_time)
