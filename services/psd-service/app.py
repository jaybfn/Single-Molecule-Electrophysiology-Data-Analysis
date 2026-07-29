"""Power spectral density microservice."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from pynanopore import CompositePSDFitter, LorentzianFitter, PSDAnalyzer, load_trace
from pynanopore.viz import plot_psd

app = FastAPI(
    title="Pynanopore PSD Service",
    version="2.3.0",
    description="Welch PSD estimation with Lorentzian / composite fits.",
)


class PSDArrayRequest(BaseModel):
    current: list[float]
    fs: float = Field(..., gt=0)
    fit: bool = True
    fit_model: Literal["lorentzian", "composite", "none"] = "lorentzian"
    include_plot: bool = False
    max_frequency: float = Field(10000.0, gt=0)
    nperseg: int | None = Field(None, gt=0)
    noverlap: int | None = Field(None, ge=0)
    window: str = "hamming"
    scaling: Literal["density", "spectrum"] = "spectrum"
    skip_bins: int = Field(2, ge=0)


class PSDResponse(BaseModel):
    request_id: str
    fs: float
    n_frequencies: int
    frequencies: list[float]
    power_spectrum: list[float]
    fit_model: str | None = None
    S0: float | None = None
    fc: float | None = None
    A: float | None = None
    alpha: float | None = None
    diagnostics: dict[str, Any] | None = None
    plot: dict[str, Any] | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "psd-service"}


def _analyze(
    current,
    fs: float,
    *,
    fit: bool,
    fit_model: str,
    include_plot: bool,
    max_frequency: float,
    nperseg: int | None,
    noverlap: int | None,
    window: str,
    scaling: str,
    skip_bins: int,
) -> PSDResponse:
    request_id = str(uuid4())
    analyzer = PSDAnalyzer(fs=fs)
    frequencies, power_spectrum = analyzer.compute_psd(
        current,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,  # type: ignore[arg-type]
        scaling=scaling,  # type: ignore[arg-type]
        skip_bins=skip_bins,
    )

    s0 = fc = a = alpha = None
    diagnostics = None
    fitter = None
    model_used: str | None = None

    if fit and fit_model != "none":
        model_used = fit_model
        if fit_model == "composite":
            fitter = CompositePSDFitter(frequencies, power_spectrum, max_frequency=max_frequency)
            params = fitter.fit()
            s0, fc, a, alpha = params["S0"], params["fc"], params["A"], params["alpha"]
        else:
            fitter = LorentzianFitter(frequencies, power_spectrum, max_frequency=max_frequency)
            s0, fc = fitter.fit_lorentzian()
        if fitter.diagnostics is not None:
            diagnostics = fitter.diagnostics.to_dict()

    plot_payload = None
    if include_plot:
        fig = plot_psd(frequencies, power_spectrum, fitter=fitter, max_freq=fs)
        plot_payload = fig.to_plotly_json()

    return PSDResponse(
        request_id=request_id,
        fs=fs,
        n_frequencies=len(frequencies),
        frequencies=frequencies.tolist(),
        power_spectrum=power_spectrum.tolist(),
        fit_model=model_used,
        S0=s0,
        fc=fc,
        A=a,
        alpha=alpha,
        diagnostics=diagnostics,
        plot=plot_payload,
    )


@app.post("/v1/psd", response_model=PSDResponse)
def compute_psd_from_array(body: PSDArrayRequest) -> PSDResponse:
    try:
        return _analyze(
            np.asarray(body.current, dtype=float),
            body.fs,
            fit=body.fit,
            fit_model=body.fit_model,
            include_plot=body.include_plot,
            max_frequency=body.max_frequency,
            nperseg=body.nperseg,
            noverlap=body.noverlap,
            window=body.window,
            scaling=body.scaling,
            skip_bins=body.skip_bins,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/psd/upload", response_model=PSDResponse)
async def compute_psd_from_file(
    file: UploadFile = File(...),
    fs: float | None = Query(None, gt=0),
    fit: bool = Query(True),
    fit_model: Literal["lorentzian", "composite", "none"] = Query("lorentzian"),
    include_plot: bool = Query(False),
    max_frequency: float = Query(10000.0, gt=0),
    nperseg: int | None = Query(None, gt=0),
    noverlap: int | None = Query(None, ge=0),
    window: str = Query("hamming"),
    scaling: Literal["density", "spectrum"] = Query("spectrum"),
    skip_bins: int = Query(2, ge=0),
) -> PSDResponse:
    suffix = Path(file.filename or "upload.abf").suffix.lower()
    if suffix not in {".abf", ".csv"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
    try:
        raw = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        trace = load_trace(tmp_path)
        sample_rate = fs if fs is not None else trace.sample_rate
        return _analyze(
            trace.current,
            sample_rate,
            fit=fit,
            fit_model=fit_model,
            include_plot=include_plot,
            max_frequency=max_frequency,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
            scaling=scaling,
            skip_bins=skip_bins,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if "tmp_path" in locals() and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
