"""Batch processing pipelines for multi-file analysis."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from pynanopore._version import __version__
from pynanopore.detection.baseline import (
    ConstantBaseline,
    MedianBaseline,
    NoneBaseline,
    PercentileBaseline,
)
from pynanopore.detection.events import EventDetector
from pynanopore.dwelltime.fit import DwellTimeExponentialFit
from pynanopore.io.readers import load_trace

SCHEMA_VERSION = "1.1.0"
SUPPORTED_SUFFIXES = {".abf", ".csv"}


@dataclass
class BatchDetectConfig:
    std_multiplier: float = 0.25
    threshold_multiplier: float = 1.5
    min_duration: float = 1e-4
    direction: Literal["down", "up"] = "down"
    baseline: Literal["none", "median", "constant", "percentile"] = "none"
    baseline_window: float = 0.05
    baseline_percentile: float = 90.0
    interval_length: float = 5.0
    overlap: float = 0.0
    fit_dwelltime: bool = True
    dwell_fit_type: Literal["single", "double", "auto"] = "single"
    analyze_levels: bool = True
    n_jobs: int = 1


def _baseline_from_name(name: str, window_s: float, percentile: float = 90.0):
    if name == "median":
        return MedianBaseline(window_s=window_s)
    if name == "constant":
        return ConstantBaseline()
    if name == "percentile":
        return PercentileBaseline(percentile=percentile, window_s=max(window_s, 0.5))
    return NoneBaseline()


def discover_recordings(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    files = sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    return files


def _process_one_file(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker for parallel batch (must be top-level for ProcessPool pickling)."""
    path = Path(payload["path"])
    out_dir = Path(payload["out_dir"])
    events_dir = out_dir / "events"
    cfg = BatchDetectConfig(**payload["config"])

    try:
        detector = EventDetector(
            std_multiplier=cfg.std_multiplier,
            threshold_multiplier=cfg.threshold_multiplier,
            min_duration=cfg.min_duration,
            direction=cfg.direction,
            baseline=_baseline_from_name(
                cfg.baseline, cfg.baseline_window, cfg.baseline_percentile
            ),
            analyze_levels=cfg.analyze_levels,
        )
        trace = load_trace(path)
        events = detector.detect_trace(
            trace, interval_length=cfg.interval_length, overlap=cfg.overlap
        )
        df = pd.DataFrame([e.to_dict() for e in events])
        out_csv = events_dir / f"{path.stem}_events.csv"
        df.to_csv(out_csv, index=False)

        row: dict[str, Any] = {
            "file": path.name,
            "status": "ok",
            "n_events": len(events),
            "sample_rate": trace.sample_rate,
            "duration_s": trace.duration,
            "median_dwell": float(df["difference"].median()) if len(df) else None,
            "median_delta_i_over_i0": (
                float(df["delta_i_over_i0"].median())
                if len(df) and "delta_i_over_i0" in df.columns
                else None
            ),
            "median_area": float(df["area"].median()) if len(df) and "area" in df.columns else None,
            "frac_multilevel": (
                float((df["n_levels"] > 1).mean()) if len(df) and "n_levels" in df.columns else None
            ),
            "events_csv": str(out_csv.relative_to(out_dir)),
            "error": None,
        }
        if cfg.fit_dwelltime and len(df) >= 5:
            try:
                fit = DwellTimeExponentialFit(df, bins=min(50, max(10, len(df) // 5)))
                result = fit.fit(cfg.dwell_fit_type, method="mle")
                row["dwell_fit_type"] = result.fit_type
                row["dwell_aic"] = result.aic
                for k, v in result.parameters.items():
                    row[f"dwell_{k}"] = v
            except Exception as fit_exc:  # noqa: BLE001
                row["dwell_fit_error"] = str(fit_exc)
        return row
    except Exception as exc:  # noqa: BLE001
        return {
            "file": path.name,
            "status": "error",
            "n_events": 0,
            "error": str(exc),
        }


def batch_detect(
    input_dir: str | Path,
    output_dir: str | Path,
    config: BatchDetectConfig | None = None,
) -> pd.DataFrame:
    """
    Run event detection on all ABF/CSV files in ``input_dir``.

    Writes:
    - ``output_dir/events/<stem>_events.csv``
    - ``output_dir/summary.csv``
    - ``output_dir/run_metadata.json``

    Set ``config.n_jobs > 1`` (or ``-1`` for all CPUs) to process files in parallel.
    """
    cfg = config or BatchDetectConfig()
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    events_dir = out_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)

    files = discover_recordings(in_dir)
    if not files:
        raise FileNotFoundError(f"No .abf/.csv files found in {in_dir}")

    n_jobs = int(cfg.n_jobs)
    if n_jobs == -1:
        import os

        n_jobs = max(1, os.cpu_count() or 1)
    n_jobs = max(1, n_jobs)

    cfg_dict = asdict(cfg)
    payloads = [
        {"path": str(p.resolve()), "out_dir": str(out_dir.resolve()), "config": cfg_dict}
        for p in files
    ]

    rows: list[dict[str, Any]] = []
    if n_jobs == 1:
        for payload in payloads:
            rows.append(_process_one_file(payload))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {pool.submit(_process_one_file, p): p for p in payloads}
            for fut in as_completed(futures):
                rows.append(fut.result())

    # Stable summary order by filename
    rows.sort(key=lambda r: str(r.get("file", "")))
    summary = pd.DataFrame(rows)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "pynanopore_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(in_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
        "n_files": len(files),
        "n_jobs": n_jobs,
        "config": cfg_dict,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary
