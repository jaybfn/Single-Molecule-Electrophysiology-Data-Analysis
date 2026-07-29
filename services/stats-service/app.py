"""Dwell-time / statistical analysis microservice."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pynanopore import DwellTimeExponentialFit
from pynanopore.viz import plot_dwelltime_histogram

app = FastAPI(
    title="Pynanopore Stats Service",
    version="2.0.0",
    description="Dwell-time histogram and exponential fitting.",
)


class StatsRequest(BaseModel):
    events: list[dict[str, float]]
    bins: int = Field(50, ge=1, le=10000)
    fit_type: Literal["single", "double"] = "single"
    percentile_clip: float = Field(99.9, gt=0, le=100)
    include_plot: bool = False


class StatsResponse(BaseModel):
    request_id: str
    n_events: int
    n_events_used: int
    fit_type: str
    parameters: dict[str, float]
    bin_centers: list[float]
    hist: list[float]
    fitted: list[float]
    plot: dict[str, Any] | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "stats-service"}


@app.post("/v1/dwelltime", response_model=StatsResponse)
def fit_dwelltime(body: StatsRequest) -> StatsResponse:
    request_id = str(uuid4())
    if not body.events:
        raise HTTPException(status_code=400, detail="events list is empty")

    try:
        df = pd.DataFrame(body.events)
        if "difference" not in df.columns:
            raise HTTPException(status_code=400, detail="events must include 'difference'")

        if 0 < body.percentile_clip < 100:
            clip = float(df["difference"].quantile(body.percentile_clip / 100.0))
            df = df[df["difference"] < clip]

        if df.empty:
            raise HTTPException(
                status_code=400, detail="No events remain after percentile clipping"
            )

        fitter = DwellTimeExponentialFit(df, bins=body.bins)
        fitter.fit_data(body.fit_type)
        params_tuple = fitter.get_parameters(body.fit_type)

        if body.fit_type == "single":
            parameters = {"a": params_tuple[0], "tau": params_tuple[1]}
        else:
            parameters = {
                "a1": params_tuple[0],
                "tau1": params_tuple[1],
                "a2": params_tuple[2],
                "tau2": params_tuple[3],
            }

        plot_payload = None
        if body.include_plot:
            fig = plot_dwelltime_histogram(fitter, fit_type=body.fit_type)
            plot_payload = fig.to_plotly_json()

        return StatsResponse(
            request_id=request_id,
            n_events=len(body.events),
            n_events_used=len(df),
            fit_type=body.fit_type,
            parameters=parameters,
            bin_centers=fitter.bin_centers.tolist(),
            hist=fitter.hist.tolist(),
            fitted=fitter.fitted_curve(body.fit_type).tolist(),
            plot=plot_payload,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
