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
    PulseShapeIdealizer,
    load_trace,
)
from pynanopore.serving import ServiceSettings, configure_service
from pynanopore.serving.app_factory import enforce_upload_size
from pynanopore.viz import Plotting, plot_pulse_shape

settings = ServiceSettings(service_name="event-service")

app = FastAPI(
    title="Pynanopore Event Service",
    version="2.4.0",
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
    preview_time: list[float] | None = None
    preview_current: list[float] | None = None


def _make_baseline(kind: str, window_s: float):
    if kind == "median":
        return MedianBaseline(window_s=window_s)
    if kind == "constant":
        return ConstantBaseline()
    return NoneBaseline()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "event-service"}


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
    baseline: Literal["none", "median", "constant"] = Query("none"),
    baseline_window: float = Query(0.05, gt=0),
    max_plot_points: int = Query(50000, ge=100),
    include_plot: bool = Query(False),
    include_pulse_plot: bool = Query(True),
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

        trace = load_trace(tmp_path)
        detector = EventDetector(
            std_multiplier=std_multiplier,
            threshold_multiplier=threshold_multiplier,
            min_duration=min_duration,
            direction=direction,
            baseline=_make_baseline(baseline, baseline_window),
        )
        events = detector.detect_trace(trace, interval_length=interval_length, overlap=overlap)
        event_dicts = [e.to_dict() for e in events]

        n = len(trace.time)
        end_idx = min(n, max_plot_points)
        if events:
            last_event = events[min(len(events), 100) - 1]
            approx_idx = int(last_event.end_time * trace.sample_rate) + int(
                0.02 * trace.sample_rate
            )
            end_idx = min(n, max(end_idx, approx_idx))

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
        if include_pulse_plot and events:
            from dataclasses import replace

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

        return DetectResponse(
            request_id=request_id,
            n_events=len(event_dicts),
            sample_rate=trace.sample_rate,
            duration_s=trace.duration,
            events=event_dicts,
            plot=plot_payload,
            pulse_plot=pulse_plot,
            preview_time=preview_time,
            preview_current=preview_current,
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
