"""Event detection microservice."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from pynanopore import EventDetector, load_trace
from pynanopore.viz import Plotting

app = FastAPI(
    title="Pynanopore Event Service",
    version="2.0.0",
    description="Detect translocation events in ABF/CSV ion-current recordings.",
)


class DetectResponse(BaseModel):
    request_id: str
    n_events: int
    sample_rate: float
    duration_s: float
    events: list[dict[str, float]]
    plot: dict[str, Any] | None = None
    preview_time: list[float] | None = None
    preview_current: list[float] | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "event-service"}


@app.post("/v1/detect", response_model=DetectResponse)
async def detect_events(
    file: UploadFile = File(...),
    std_multiplier: float = Query(0.25, ge=0),
    threshold_multiplier: float = Query(1.5, ge=0),
    interval_length: float = Query(5.0, gt=0),
    min_duration: float = Query(1e-4, ge=0),
    max_plot_points: int = Query(50000, ge=100),
    include_plot: bool = Query(False),
) -> DetectResponse:
    request_id = str(uuid4())
    suffix = Path(file.filename or "upload.abf").suffix.lower()
    if suffix not in {".abf", ".csv"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    try:
        raw = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)

        trace = load_trace(tmp_path)
        detector = EventDetector(
            std_multiplier=std_multiplier,
            threshold_multiplier=threshold_multiplier,
            min_duration=min_duration,
        )
        events = detector.detect_trace(trace, interval_length=interval_length)
        event_dicts = [e.to_dict() for e in events]

        # Preview: up to max_plot_points, covering early events when present
        n = len(trace.time)
        end_idx = min(n, max_plot_points)
        if events:
            # Extend preview to cover a bit past the last shown event without hardcoding [100]
            last_event = events[min(len(events), 100) - 1] if events else None
            if last_event is not None:
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

        return DetectResponse(
            request_id=request_id,
            n_events=len(event_dicts),
            sample_rate=trace.sample_rate,
            duration_s=trace.duration,
            events=event_dicts,
            plot=plot_payload,
            preview_time=preview_time,
            preview_current=preview_current,
        )
    except Exception as exc:  # noqa: BLE001 — surface as HTTP error
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if "tmp_path" in locals() and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.post("/v1/detect/from-events-frame")
def events_summary(events: list[dict[str, float]]) -> dict[str, Any]:
    df = pd.DataFrame(events)
    return {"n_events": len(df), "columns": list(df.columns)}
