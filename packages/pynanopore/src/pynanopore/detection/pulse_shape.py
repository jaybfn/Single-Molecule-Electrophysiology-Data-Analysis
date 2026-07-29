"""Idealized pulse-shape reconstruction from detected events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pynanopore.detection.events import Event
from pynanopore.io.trace import Trace


@dataclass
class PulseShapeResult:
    """Idealized stepwise current matching detected events."""

    time: NDArray[np.floating]
    idealized: NDArray[np.floating]
    open_level: NDArray[np.floating]
    events: list[Event]

    @property
    def rising_edges(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Times and currents at event starts (rising into the blocked state)."""
        if not self.events:
            return np.array([]), np.array([])
        t = np.array([e.start_time for e in self.events], dtype=float)
        y = np.array([e.blockade_mean for e in self.events], dtype=float)
        return t, y

    @property
    def falling_edges(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Times and currents at event ends (return to open pore)."""
        if not self.events:
            return np.array([]), np.array([])
        t = np.array([e.end_time for e in self.events], dtype=float)
        y = np.array([e.i0 for e in self.events], dtype=float)
        return t, y


class PulseShapeIdealizer:
    """
    Build a rectangular pulse idealization from events (Clampfit/QuB-style).

    Outside events the idealized trace equals the open-pore level ``i0``
    (per-event local baseline at the event, or a global open level).
    Inside each event it equals ``blockade_mean``.
    """

    def __init__(self, *, use_event_i0: bool = True, global_open_level: float | None = None):
        self.use_event_i0 = use_event_i0
        self.global_open_level = global_open_level

    @classmethod
    def from_events(
        cls,
        trace: Trace,
        events: list[Event],
        *,
        use_event_i0: bool = True,
        global_open_level: float | None = None,
    ) -> PulseShapeResult:
        return cls(use_event_i0=use_event_i0, global_open_level=global_open_level).idealize(
            trace, events
        )

    def idealize(self, trace: Trace, events: list[Event]) -> PulseShapeResult:
        n = len(trace.current)
        if self.global_open_level is not None:
            open_fill = float(self.global_open_level)
        elif events:
            open_fill = float(np.median([e.i0 for e in events]))
        else:
            open_fill = float(np.median(trace.current))

        idealized: NDArray[np.floating] = np.full(n, open_fill, dtype=float)
        open_level: NDArray[np.floating] = np.full(n, open_fill, dtype=float)

        for ev in events:
            start = ev.start_idx if ev.start_idx >= 0 else int(ev.start_time * trace.sample_rate)
            end = ev.end_idx if ev.end_idx >= 0 else int(ev.end_time * trace.sample_rate)
            start = max(0, min(start, n - 1))
            end = max(start, min(end, n - 1))

            if self.use_event_i0:
                open_level[start : end + 1] = ev.i0
                # Fill open regions between events with each event's i0 near edges
                # Keep global fill elsewhere; optionally paint local i0 in a small halo
            idealized[start : end + 1] = ev.blockade_mean

        return PulseShapeResult(
            time=np.asarray(trace.time, dtype=float),
            idealized=idealized,
            open_level=open_level,
            events=list(events),
        )
