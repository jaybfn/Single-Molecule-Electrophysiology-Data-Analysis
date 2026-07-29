"""API gateway / BFF for Pynanopore microservices."""

from __future__ import annotations

from typing import Any, Literal

import httpx
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

app = FastAPI(
    title="Pynanopore Gateway",
    version="2.0.0",
    description="Single entrypoint routing to event, stats, and PSD services.",
)


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    event_service_url: str = "http://event-service:8001"
    stats_service_url: str = "http://stats-service:8002"
    psd_service_url: str = "http://psd-service:8003"
    http_timeout_s: float = 120.0


settings = GatewaySettings()


class StatsProxyRequest(BaseModel):
    events: list[dict[str, float]]
    bins: int = Field(50, ge=1)
    fit_type: Literal["single", "double", "auto"] = "single"
    method: Literal["mle", "histogram"] = "mle"
    binning: Literal["linear", "log"] = "linear"
    percentile_clip: float = 99.9
    include_plot: bool = False


class PSDProxyRequest(BaseModel):
    current: list[float]
    fs: float = Field(..., gt=0)
    fit: bool = True
    include_plot: bool = False
    max_frequency: float = 10000.0


@app.get("/health")
async def health() -> dict[str, Any]:
    statuses: dict[str, Any] = {"gateway": "ok"}
    targets = {
        "event": f"{settings.event_service_url}/health",
        "stats": f"{settings.stats_service_url}/health",
        "psd": f"{settings.psd_service_url}/health",
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in targets.items():
            try:
                resp = await client.get(url)
                statuses[name] = resp.json() if resp.status_code == 200 else {"status": "degraded"}
            except httpx.HTTPError:
                statuses[name] = {"status": "down"}
    overall = (
        "ok"
        if all(
            isinstance(v, dict) and v.get("status") == "ok"
            for k, v in statuses.items()
            if k != "gateway"
        )
        else "degraded"
    )
    return {"status": overall, "services": statuses}


@app.post("/v1/detect")
async def detect(
    file: UploadFile = File(...),
    std_multiplier: float = Query(0.25),
    threshold_multiplier: float = Query(1.5),
    interval_length: float = Query(5.0),
    overlap: float = Query(0.0),
    min_duration: float = Query(1e-4),
    direction: Literal["down", "up"] = Query("down"),
    baseline: Literal["none", "median", "constant"] = Query("none"),
    baseline_window: float = Query(0.05),
    max_plot_points: int = Query(50000),
    include_plot: bool = Query(False),
    include_pulse_plot: bool = Query(True),
) -> Any:
    data = await file.read()
    files = {
        "file": (
            file.filename or "upload.abf",
            data,
            file.content_type or "application/octet-stream",
        )
    }
    params = {
        "std_multiplier": std_multiplier,
        "threshold_multiplier": threshold_multiplier,
        "interval_length": interval_length,
        "overlap": overlap,
        "min_duration": min_duration,
        "direction": direction,
        "baseline": baseline,
        "baseline_window": baseline_window,
        "max_plot_points": max_plot_points,
        "include_plot": include_plot,
        "include_pulse_plot": include_pulse_plot,
    }
    async with httpx.AsyncClient(timeout=settings.http_timeout_s) as client:
        try:
            resp = await client.post(
                f"{settings.event_service_url}/v1/detect",
                files=files,
                params=params,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502, detail=f"event-service unreachable: {exc}"
            ) from exc
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@app.post("/v1/dwelltime")
async def dwelltime(body: StatsProxyRequest) -> Any:
    async with httpx.AsyncClient(timeout=settings.http_timeout_s) as client:
        try:
            resp = await client.post(
                f"{settings.stats_service_url}/v1/dwelltime",
                json=body.model_dump(),
            )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502, detail=f"stats-service unreachable: {exc}"
            ) from exc
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@app.post("/v1/psd")
async def psd(body: PSDProxyRequest) -> Any:
    async with httpx.AsyncClient(timeout=settings.http_timeout_s) as client:
        try:
            resp = await client.post(
                f"{settings.psd_service_url}/v1/psd",
                json=body.model_dump(),
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"psd-service unreachable: {exc}") from exc
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@app.post("/v1/psd/upload")
async def psd_upload(
    file: UploadFile = File(...),
    fs: float | None = Query(None),
    fit: bool = Query(True),
    include_plot: bool = Query(False),
    max_frequency: float = Query(10000.0),
) -> Any:
    data = await file.read()
    files = {
        "file": (
            file.filename or "upload.abf",
            data,
            file.content_type or "application/octet-stream",
        )
    }
    params: dict[str, Any] = {
        "fit": fit,
        "include_plot": include_plot,
        "max_frequency": max_frequency,
    }
    if fs is not None:
        params["fs"] = fs
    async with httpx.AsyncClient(timeout=settings.http_timeout_s) as client:
        try:
            resp = await client.post(
                f"{settings.psd_service_url}/v1/psd/upload",
                files=files,
                params=params,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"psd-service unreachable: {exc}") from exc
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Service"] = "gateway"
    return response
