"""Event detection microservice."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel

from pynanopore import (
    ConstantBaseline,
    EventDetector,
    MedianBaseline,
    NoneBaseline,
    PercentileBaseline,
    PulseShapeIdealizer,
    load_trace,
)
from pynanopore.serving import ServiceSettings, configure_service
from pynanopore.serving.app_factory import enforce_upload_size
from pynanopore.viz import Plotting, plot_multi_level, plot_pulse_shape

settings = ServiceSettings(service_name="event-service")

app = FastAPI(
    title="Pynanopore Event Service",
    version="2.7.1",
    description="Detect translocation events in ABF/CSV ion-current recordings.",
)
configure_service(app, settings)


class DetectResponse(BaseModel):
    request_id: str
    n_events: int
    sample_rate: float
    duration_s: float
    events: list[dict[str, float]]
    plot: dict[str, Any] | None = None
    pulse_plot: dict[str, Any] | None = None
    levels_plot: dict[str, Any] | None = None
    preview_time: list[float] | None = None
    preview_current: list[float] | None = None
    window_start_s: float | None = None
    window_end_s: float | None = None


class PreviewResponse(BaseModel):
    request_id: str
    sample_rate: float
    duration_s: float
    t_min: float
    t_max: float
    n_points_total: int
    n_points_returned: int
    time: list[float]
    current: list[float]
    filename: str | None = None


def _make_baseline(kind: str, window_s: float, percentile: float = 90.0):
    if kind == "median":
        return MedianBaseline(window_s=window_s)
    if kind == "constant":
        return ConstantBaseline()
    if kind == "percentile":
        return PercentileBaseline(percentile=percentile, window_s=max(window_s, 0.5))
    return NoneBaseline()


def _downsample(time, current, max_points: int):
    n = len(time)
    if n <= max_points:
        return time, current
    step = max(1, n // max_points)
    return time[::step], current[::step]


def _apply_window(trace, t_start: float | None, t_end: float | None):
    if t_start is None and t_end is None:
        return trace
    return trace.slice_by_time(t_start, t_end)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "event-service"}


@app.post("/v1/preview", response_model=PreviewResponse)
async def preview_trace(
    request: Request,
    file: UploadFile = File(...),
    max_points: int = Query(20000, ge=100, le=200000),
) -> PreviewResponse:
    """Load a recording and return a downsampled preview for UI threshold tuning."""
    request_id = getattr(request.state, "request_id", "unknown")
    suffix = Path(file.filename or "upload.abf").suffix.lower()
    if suffix not in {".abf", ".csv"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
    try:
        raw = await file.read()
        enforce_upload_size(raw, settings, request_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        trace = load_trace(tmp_path)
        t, c = _downsample(trace.time, trace.current, max_points)
        return PreviewResponse(
            request_id=request_id,
            sample_rate=float(trace.sample_rate),
            duration_s=float(trace.duration),
            t_min=float(trace.time[0]),
            t_max=float(trace.time[-1]),
            n_points_total=len(trace.time),
            n_points_returned=len(t),
            time=t.tolist(),
            current=c.tolist(),
            filename=file.filename,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if "tmp_path" in locals() and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.post("/v1/detect", response_model=DetectResponse)
async def detect_events(
    request: Request,
    file: UploadFile = File(...),
    std_multiplier: float = Query(0.25, ge=0),
    threshold_multiplier: float = Query(1.5, ge=0),
    interval_length: float = Query(5.0, gt=0),
    overlap: float = Query(0.0, ge=0),
    min_duration: float = Query(1e-4, ge=0),
    direction: Literal["down", "up"] = Query("down"),
    baseline: Literal["none", "median", "constant", "percentile"] = Query("none"),
    baseline_window: float = Query(0.05, gt=0),
    baseline_percentile: float = Query(90.0, ge=0, le=100),
    max_plot_points: int = Query(50000, ge=100),
    include_plot: bool = Query(False),
    include_pulse_plot: bool = Query(True),
    analyze_levels: bool = Query(True),
    t_start: float | None = Query(None, description="Analysis window start (s)"),
    t_end: float | None = Query(None, description="Analysis window end (s)"),
) -> DetectResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    suffix = Path(file.filename or "upload.abf").suffix.lower()
    if suffix not in {".abf", ".csv"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    try:
        raw = await file.read()
        enforce_upload_size(raw, settings, request_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)

        full = load_trace(tmp_path)
        trace = _apply_window(full, t_start, t_end)
        window_start = float(trace.time[0])
        window_end = float(trace.time[-1])

        detector = EventDetector(
            std_multiplier=std_multiplier,
            threshold_multiplier=threshold_multiplier,
            min_duration=min_duration,
            direction=direction,
            baseline=_make_baseline(baseline, baseline_window, baseline_percentile),
            analyze_levels=analyze_levels,
        )
        events = detector.detect_trace(trace, interval_length=interval_length, overlap=overlap)
        event_dicts = [e.to_dict() for e in events]

        n = len(trace.time)
        end_idx = min(n, max_plot_points)
        if events:
            # Prefer last event sample index over absolute-time heuristics
            last_event = events[min(len(events), 100) - 1]
            if last_event.end_idx >= 0:
                end_idx = min(n, max(end_idx, last_event.end_idx + int(0.02 * trace.sample_rate)))

        preview_time = trace.time[:end_idx].tolist()
        preview_current = trace.current[:end_idx].tolist()

        plot_payload = None
        if include_plot:
            fig = Plotting.plot_data(
                trace.time[:end_idx],
                trace.current[:end_idx],
                event_dicts,
                std_multiplier=std_multiplier,
                threshold_multiplier=threshold_multiplier,
            )
            plot_payload = fig.to_plotly_json()

        pulse_plot = None
        levels_plot = None
        if include_pulse_plot and events:
            from dataclasses import replace

            from pynanopore.detection.levels import idealize_multilevel
            from pynanopore.io.trace import Trace

            preview_trace = Trace(
                time=trace.time[:end_idx],
                current=trace.current[:end_idx],
                sample_rate=trace.sample_rate,
                source=trace.source,
            )
            preview_events = []
            for e in events:
                if e.start_idx >= end_idx:
                    continue
                end_i = min(e.end_idx if e.end_idx >= 0 else end_idx - 1, end_idx - 1)
                start_i = max(0, e.start_idx)
                if end_i >= start_i:
                    preview_events.append(replace(e, start_idx=start_i, end_idx=end_i))
            pulse = PulseShapeIdealizer.from_events(preview_trace, preview_events)
            pulse_fig = plot_pulse_shape(preview_trace.time, preview_trace.current, pulse)
            pulse_plot = pulse_fig.to_plotly_json()

            if analyze_levels and preview_events:
                multilevel = idealize_multilevel(preview_trace, preview_events)
                levels_fig = plot_multi_level(preview_trace.time, preview_trace.current, multilevel)
                levels_plot = levels_fig.to_plotly_json()

        return DetectResponse(
            request_id=request_id,
            n_events=len(event_dicts),
            sample_rate=trace.sample_rate,
            duration_s=trace.duration,
            events=event_dicts,
            plot=plot_payload,
            pulse_plot=pulse_plot,
            levels_plot=levels_plot,
            preview_time=preview_time,
            preview_current=preview_current,
            window_start_s=window_start,
            window_end_s=window_end,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if "tmp_path" in locals() and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.post("/v1/detect/from-events-frame")
def events_summary(events: list[dict[str, float]]) -> dict[str, Any]:
    df = pd.DataFrame(events)
    return {"n_events": len(df), "columns": list(df.columns)}
