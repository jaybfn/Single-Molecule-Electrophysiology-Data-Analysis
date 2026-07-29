"""Threshold-based event detection on ion-current chunks."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pynanopore.detection.baseline import BaselineEstimator, NoneBaseline, residual_current
from pynanopore.io.trace import Trace

EventDirection = Literal["down", "up"]


@dataclass
class Event:
    """A detected translocation / blockade event with rich features."""

    start_time: float
    end_time: float
    difference: float  # dwell time (s); kept for backward compatibility
    amplitude: float  # extreme current in the event (min for down, max for up)

    # Rich features (Phase A)
    dwell_time: float = 0.0
    i0: float = 0.0
    blockade_mean: float = 0.0
    blockade_min: float = 0.0
    blockade_max: float = 0.0
    delta_i: float = 0.0
    delta_i_over_i0: float = 0.0
    area: float = 0.0
    rise_time: float = 0.0
    fall_time: float = 0.0
    start_idx: int = -1
    end_idx: int = -1

    def __post_init__(self) -> None:
        if self.dwell_time == 0.0 and self.difference != 0.0:
            self.dwell_time = self.difference

    def to_dict(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.generic):
                v = v.item()
            out[f.name] = float(v)
        return out


def _transition_times(
    segment: NDArray[np.floating],
    time_segment: NDArray[np.floating],
    i0: float,
    ib: float,
) -> tuple[float, float]:
    """Estimate 10–90% rise and fall times within an event segment."""
    if len(segment) < 3:
        return 0.0, 0.0
    depth = ib - i0
    if abs(depth) < 1e-12:
        return 0.0, 0.0

    frac = (segment - i0) / depth
    # Rise: first time crossing 0.1 then 0.9
    rise_lo = np.where(frac >= 0.1)[0]
    rise_hi = np.where(frac >= 0.9)[0]
    rise = 0.0
    if len(rise_lo) and len(rise_hi) and rise_hi[0] >= rise_lo[0]:
        rise = float(time_segment[rise_hi[0]] - time_segment[rise_lo[0]])

    # Fall: from end, crossing 0.9 then 0.1 (leaving blocked state)
    rev = frac[::-1]
    t_rev = time_segment[::-1]
    fall_hi = np.where(rev >= 0.9)[0]
    fall_lo = np.where(rev <= 0.1)[0]
    fall = 0.0
    if len(fall_hi) and len(fall_lo):
        # indices in reversed array
        # find first rev>=0.9 from start of reverse (= end of event), then first <=0.1 after that
        hi_idx = fall_hi[0]
        lo_candidates = fall_lo[fall_lo >= hi_idx]
        if len(lo_candidates):
            fall = abs(float(t_rev[lo_candidates[0]] - t_rev[hi_idx]))

    return max(0.0, rise), max(0.0, fall)


def _build_event(
    *,
    current: NDArray[np.floating],
    time: NDArray[np.floating],
    start_idx: int,
    end_idx: int,
    direction: EventDirection,
    sample_rate: float,
    baseline_value: float,
) -> Event:
    segment = current[start_idx : end_idx + 1]
    time_segment = time[start_idx : end_idx + 1]
    start_time = float(time[start_idx])
    end_time = float(time[end_idx])
    dwell = end_time - start_time

    i0 = float(baseline_value)
    blockade_mean = float(np.mean(segment))
    blockade_min = float(np.min(segment))
    blockade_max = float(np.max(segment))
    amplitude = blockade_min if direction == "down" else blockade_max
    delta_i = abs(i0 - blockade_mean)
    delta_i_over_i0 = delta_i / abs(i0) if abs(i0) > 1e-12 else float("nan")

    # Area: integral of |I0 - I| over the event (trapezoid)
    dt = (
        1.0 / sample_rate
        if sample_rate > 0
        else float(np.median(np.diff(time_segment)))
        if len(time_segment) > 1
        else 0.0
    )
    area = float(np.sum(np.abs(i0 - segment)) * dt)

    rise_time, fall_time = _transition_times(segment, time_segment, i0, blockade_mean)

    return Event(
        start_time=start_time,
        end_time=end_time,
        difference=dwell,
        amplitude=amplitude,
        dwell_time=dwell,
        i0=i0,
        blockade_mean=blockade_mean,
        blockade_min=blockade_min,
        blockade_max=blockade_max,
        delta_i=delta_i,
        delta_i_over_i0=delta_i_over_i0,
        area=area,
        rise_time=rise_time,
        fall_time=fall_time,
        start_idx=int(start_idx),
        end_idx=int(end_idx),
    )


class EventDetector:
    """
    Dual-threshold event detector with optional baseline correction and polarity.

    Detection runs on a canonical residual where events are **downward**. For
    ``direction='up'`` (pulses above baseline, as in many Axon recordings with
    negative open-pore current), the residual is sign-flipped before thresholding.
    """

    def __init__(
        self,
        std_multiplier: float = 0.25,
        threshold_multiplier: float = 1.5,
        min_duration: float = 1e-4,
        *,
        direction: EventDirection = "down",
        baseline: BaselineEstimator | None = None,
        sample_rate: float | None = None,
    ):
        if std_multiplier < 0 or threshold_multiplier < 0:
            raise ValueError("multipliers must be non-negative")
        if min_duration < 0:
            raise ValueError("min_duration must be non-negative")
        self.std_multiplier = float(std_multiplier)
        self.threshold_multiplier = float(threshold_multiplier)
        self.min_duration = float(min_duration)
        self.direction: EventDirection = direction
        self.baseline: BaselineEstimator = baseline if baseline is not None else NoneBaseline()
        self.sample_rate = sample_rate

    def _prepare_signal(
        self,
        current: NDArray[np.floating],
        sample_rate: float,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Return ``(work_signal, baseline, raw_current)`` with events downward on work_signal."""
        raw = np.asarray(current, dtype=float)
        if isinstance(self.baseline, NoneBaseline):
            baseline = np.full_like(raw, float(np.mean(raw)))
            residual = raw - baseline
        else:
            baseline = np.asarray(self.baseline.estimate(raw, sample_rate), dtype=float)
            residual = residual_current(raw, baseline)

        work = residual if self.direction == "down" else -residual
        return work, baseline, raw

    def detect_events(
        self,
        data_chunk: NDArray[np.floating],
        data_time: NDArray[np.floating],
        *,
        sample_rate: float | None = None,
        index_offset: int = 0,
    ) -> list[Event]:
        """Detect events in a single chunk; returns Event objects with rich features."""
        if len(data_chunk) < 2:
            return []
        if len(data_chunk) != len(data_time):
            raise ValueError("data_chunk and data_time must have the same length")

        fs = sample_rate or self.sample_rate
        if fs is None:
            diffs = np.diff(np.asarray(data_time, dtype=float))
            fs = float(1.0 / np.median(diffs)) if len(diffs) and np.median(diffs) > 0 else 1.0

        work, baseline, raw = self._prepare_signal(data_chunk, fs)
        mean = float(np.mean(work))
        std_dev = float(np.std(work))
        if std_dev == 0:
            return []

        # On the canonical (downward) work signal, thresholds are below the mean
        entry = mean - self.std_multiplier * std_dev
        deep = mean - self.threshold_multiplier * std_dev

        events: list[Event] = []
        start_time: float | None = None
        start_idx: int | None = None
        crossed_deep = False

        for i in range(1, len(work)):
            if work[i] < entry and work[i - 1] >= entry:
                start_time = float(data_time[i])
                start_idx = i
                crossed_deep = False

            if start_time is not None and work[i] < deep:
                crossed_deep = True

            if (
                start_time is not None
                and start_idx is not None
                and work[i] >= entry
                and work[i - 1] < entry
            ):
                if crossed_deep:
                    end_idx = i
                    duration = float(data_time[end_idx]) - start_time
                    if duration >= self.min_duration:
                        # Local open-pore estimate: baseline at event start (or mean of nearby baseline)
                        i0_local = float(baseline[start_idx])
                        events.append(
                            _build_event(
                                current=raw,
                                time=np.asarray(data_time, dtype=float),
                                start_idx=start_idx,
                                end_idx=end_idx,
                                direction=self.direction,
                                sample_rate=fs,
                                baseline_value=i0_local,
                            )
                        )
                        # Adjust absolute indices if chunked
                        events[-1].start_idx = index_offset + start_idx
                        events[-1].end_idx = index_offset + end_idx
                start_time = None
                start_idx = None
                crossed_deep = False

        return events

    def detect_events_dicts(
        self,
        data_chunk: NDArray[np.floating],
        data_time: NDArray[np.floating],
        **kwargs,
    ) -> list[dict[str, float]]:
        """Backward-compatible list-of-dicts API."""
        return [e.to_dict() for e in self.detect_events(data_chunk, data_time, **kwargs)]

    def detect_trace(
        self,
        trace: Trace,
        *,
        interval_length: float = 5.0,
        overlap: float = 0.0,
    ) -> list[Event]:
        """Run detection over an entire Trace using fixed-length chunks.

        Parameters
        ----------
        overlap:
            Overlap between consecutive chunks in seconds (reduces edge misses).
            Events whose start falls in the overlap tail of the previous chunk
            are skipped to avoid duplicates.
        """
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= interval_length:
            raise ValueError("overlap must be smaller than interval_length")

        fs = trace.sample_rate
        self.sample_rate = fs
        step = max(1, int((interval_length - overlap) * fs))
        win = max(1, int(interval_length * fs))
        n = len(trace.current)
        all_events: list[Event] = []
        seen_starts: set[int] = set()

        for start in range(0, n, step):
            end = min(start + win, n)
            if end - start < 2:
                break
            chunk_events = self.detect_events(
                trace.current[start:end],
                trace.time[start:end],
                sample_rate=fs,
                index_offset=start,
            )
            for ev in chunk_events:
                # Deduplicate by absolute start index
                key = ev.start_idx
                if key in seen_starts:
                    continue
                seen_starts.add(key)
                all_events.append(ev)
            if end >= n:
                break

        return all_events


# Backward-compatible alias
class EventDetection(EventDetector):
    """Deprecated: prefer :class:`EventDetector`."""

    def detect_events(  # type: ignore[override]
        self,
        data_chunk: NDArray[np.floating],
        data_time: NDArray[np.floating],
        **kwargs,
    ) -> list[dict[str, float]]:
        return self.detect_events_dicts(data_chunk, data_time, **kwargs)
