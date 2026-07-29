"""Batch processing pipelines for multi-file analysis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from pynanopore._version import __version__
from pynanopore.detection.baseline import ConstantBaseline, MedianBaseline, NoneBaseline
from pynanopore.detection.events import EventDetector
from pynanopore.dwelltime.fit import DwellTimeExponentialFit
from pynanopore.io.readers import load_trace

SCHEMA_VERSION = "1.0.0"
SUPPORTED_SUFFIXES = {".abf", ".csv"}


@dataclass
class BatchDetectConfig:
    std_multiplier: float = 0.25
    threshold_multiplier: float = 1.5
    min_duration: float = 1e-4
    direction: Literal["down", "up"] = "down"
    baseline: Literal["none", "median", "constant"] = "none"
    baseline_window: float = 0.05
    interval_length: float = 5.0
    overlap: float = 0.0
    fit_dwelltime: bool = True
    dwell_fit_type: Literal["single", "double", "auto"] = "single"


def _baseline_from_name(name: str, window_s: float):
    if name == "median":
        return MedianBaseline(window_s=window_s)
    if name == "constant":
        return ConstantBaseline()
    return NoneBaseline()


def discover_recordings(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    files = sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    return files


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
    """
    cfg = config or BatchDetectConfig()
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    events_dir = out_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)

    files = discover_recordings(in_dir)
    if not files:
        raise FileNotFoundError(f"No .abf/.csv files found in {in_dir}")

    detector = EventDetector(
        std_multiplier=cfg.std_multiplier,
        threshold_multiplier=cfg.threshold_multiplier,
        min_duration=cfg.min_duration,
        direction=cfg.direction,
        baseline=_baseline_from_name(cfg.baseline, cfg.baseline_window),
    )

    rows: list[dict[str, Any]] = []
    for path in files:
        try:
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
                "median_area": float(df["area"].median())
                if len(df) and "area" in df.columns
                else None,
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
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "file": path.name,
                    "status": "error",
                    "n_events": 0,
                    "error": str(exc),
                }
            )

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
        "config": asdict(cfg),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary
