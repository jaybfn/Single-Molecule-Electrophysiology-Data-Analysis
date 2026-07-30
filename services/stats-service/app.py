"""Dwell-time / statistical analysis microservice."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from pynanopore import DwellTimeExponentialFit
from pynanopore.serving import ServiceSettings, configure_service
from pynanopore.viz import plot_dwelltime_histogram

settings = ServiceSettings(service_name="stats-service")

app = FastAPI(
    title="Pynanopore Stats Service",
    version="2.7.0",
    description="Dwell-time histogram and exponential lifetime fitting (MLE / AIC).",
)
configure_service(app, settings)


class StatsRequest(BaseModel):
    events: list[dict[str, float]]
    bins: int = Field(50, ge=1, le=10000)
    fit_type: Literal["single", "double", "auto"] = "single"
    method: Literal["mle", "histogram"] = "mle"
    binning: Literal["linear", "log"] = "linear"
    percentile_clip: float = Field(99.9, gt=0, le=100)
    include_plot: bool = False


class StatsResponse(BaseModel):
    request_id: str
    n_events: int
    n_events_used: int
    fit_type: str
    method: str
    parameters: dict[str, float]
    log_likelihood: float | None = None
    aic: float | None = None
    bic: float | None = None
    model_comparison: dict[str, Any] | None = None
    bin_centers: list[float]
    hist: list[float]
    fitted: list[float]
    plot: dict[str, Any] | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "stats-service"}


@app.post("/v1/dwelltime", response_model=StatsResponse)
def fit_dwelltime(request: Request, body: StatsRequest) -> StatsResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    if not body.events:
        raise HTTPException(status_code=400, detail="events list is empty")

    try:
        df = pd.DataFrame(body.events)
        if "difference" not in df.columns and "dwell_time" not in df.columns:
            raise HTTPException(
                status_code=400, detail="events must include 'difference' or 'dwell_time'"
            )

        dwell_col = "difference" if "difference" in df.columns else "dwell_time"
        if 0 < body.percentile_clip < 100:
            clip = float(df[dwell_col].quantile(body.percentile_clip / 100.0))
            df = df[df[dwell_col] < clip]

        if df.empty:
            raise HTTPException(
                status_code=400, detail="No events remain after percentile clipping"
            )

        fitter = DwellTimeExponentialFit(df, bins=body.bins, binning=body.binning)
        result = fitter.fit(body.fit_type, method=body.method)

        comparison = None
        if body.fit_type == "auto" or body.method == "mle":
            try:
                both = fitter.compare_models()
                comparison = {
                    name: {
                        "parameters": r.parameters,
                        "aic": r.aic,
                        "bic": r.bic,
                        "log_likelihood": r.log_likelihood,
                    }
                    for name, r in both.items()
                }
            except Exception:  # noqa: BLE001
                comparison = None

        plot_payload = None
        if body.include_plot:
            fitter.last_result = result
            fig = plot_dwelltime_histogram(fitter, fit_type=result.fit_type)
            plot_payload = fig.to_plotly_json()

        return StatsResponse(
            request_id=request_id,
            n_events=len(body.events),
            n_events_used=len(df),
            fit_type=result.fit_type,
            method=result.method,
            parameters=result.parameters,
            log_likelihood=result.log_likelihood,
            aic=result.aic,
            bic=result.bic,
            model_comparison=comparison,
            bin_centers=result.bin_centers,
            hist=result.hist,
            fitted=result.fitted,
            plot=plot_payload,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
