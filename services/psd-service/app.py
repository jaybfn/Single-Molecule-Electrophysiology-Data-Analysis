"""Power spectral density microservice."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from pynanopore import LorentzianFitter, PSDAnalyzer, load_trace
from pynanopore.viz import plot_psd

app = FastAPI(
    title="Pynanopore PSD Service",
    version="2.0.0",
    description="Welch PSD estimation and Lorentzian fitting.",
)


class PSDArrayRequest(BaseModel):
    current: list[float]
    fs: float = Field(..., gt=0)
    fit: bool = True
    include_plot: bool = False
    max_frequency: float = Field(10000.0, gt=0)


class PSDResponse(BaseModel):
    request_id: str
    fs: float
    n_frequencies: int
    frequencies: list[float]
    power_spectrum: list[float]
    S0: float | None = None
    fc: float | None = None
    plot: dict[str, Any] | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "psd-service"}


def _analyze(
    current,
    fs: float,
    *,
    fit: bool,
    include_plot: bool,
    max_frequency: float,
) -> PSDResponse:
    request_id = str(uuid4())
    analyzer = PSDAnalyzer(fs=fs)
    frequencies, power_spectrum = analyzer.compute_psd_with_hamming(current)

    s0 = fc = None
    fitter = None
    if fit:
        fitter = LorentzianFitter(frequencies, power_spectrum, max_frequency=max_frequency)
        s0, fc = fitter.fit_lorentzian()

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
        S0=s0,
        fc=fc,
        plot=plot_payload,
    )


@app.post("/v1/psd", response_model=PSDResponse)
def compute_psd_from_array(body: PSDArrayRequest) -> PSDResponse:
    try:
        import numpy as np

        return _analyze(
            np.asarray(body.current, dtype=float),
            body.fs,
            fit=body.fit,
            include_plot=body.include_plot,
            max_frequency=body.max_frequency,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/psd/upload", response_model=PSDResponse)
async def compute_psd_from_file(
    file: UploadFile = File(...),
    fs: float | None = Query(None, gt=0),
    fit: bool = Query(True),
    include_plot: bool = Query(False),
    max_frequency: float = Query(10000.0, gt=0),
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
            include_plot=include_plot,
            max_frequency=max_frequency,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if "tmp_path" in locals() and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
