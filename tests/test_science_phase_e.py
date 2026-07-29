"""Phase E science: levels, percentile baseline, multi-Lorentzian, parallel batch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pynanopore import (
    BatchDetectConfig,
    EventDetector,
    LorentzianWhiteFitter,
    MedianBaseline,
    MultiLorentzianFitter,
    PercentileBaseline,
    analyze_event_levels,
    batch_detect,
)
from pynanopore.io.trace import Trace


def test_percentile_baseline_tracks_open_pore_with_events():
    fs = 1000.0
    n = 5000
    t = np.arange(n) / fs
    # Open pore ~100 with slow drift; frequent deep events to -40
    current = 100.0 + 5.0 * np.sin(2 * np.pi * 0.2 * t)
    for start in range(200, n, 250):
        current[start : start + 40] = 40.0
    bl = PercentileBaseline(percentile=90.0, window_s=0.8).estimate(current, fs)
    # Open-pore estimate should stay near ~100, not collapse to event level
    assert float(np.median(bl)) > 90.0
    med = MedianBaseline(window_s=0.05).estimate(current, fs)
    # Percentile baseline should be higher (less pulled by down events) on average
    assert float(np.median(bl)) >= float(np.median(med)) - 1.0


def test_analyze_event_levels_two_states():
    rng = np.random.default_rng(0)
    seg = np.concatenate(
        [
            rng.normal(-20.0, 0.5, size=80),
            rng.normal(-50.0, 0.5, size=80),
        ]
    )
    feats = analyze_event_levels(seg, i0=0.0, max_levels=2, min_samples=20)
    assert feats.n_levels == 2.0
    assert abs(feats.level1_current + 50.0) < 5 or abs(feats.level1_current + 20.0) < 5
    assert feats.level1_fraction + feats.level2_fraction == pytest.approx(1.0, abs=0.05)


def test_event_detector_emits_level_fields(synthetic_trace: Trace):
    events = EventDetector(0.5, 2.0, analyze_levels=True).detect_trace(
        synthetic_trace, interval_length=1.0
    )
    assert events
    d = events[0].to_dict()
    assert "n_levels" in d
    assert "level1_current" in d


def test_lorentzian_white_fit():
    rng = np.random.default_rng(1)
    f = np.logspace(1, 3.5, 200)
    s0, fc, n = 2.0, 200.0, 0.05
    true = s0 / (1 + (f / fc) ** 2) + n
    psd = true * (1 + 0.02 * rng.normal(size=len(f)))
    fit = LorentzianWhiteFitter(f, psd, max_frequency=5000.0).fit()
    assert fit["S0"] > 0 and fit["fc"] > 0 and fit["N"] > 0


def test_multi_lorentzian_fit():
    f = np.logspace(0.5, 3.5, 250)
    true = 3.0 / (1 + (f / 50.0) ** 2) + 1.0 / (1 + (f / 800.0) ** 2) + 0.02
    fit = MultiLorentzianFitter(
        f, true, n_components=2, include_white=True, max_frequency=5000.0
    ).fit()
    assert "S0_1" in fit and "S0_2" in fit and "N" in fit
    assert fit["fc_1"] > 0 and fit["fc_2"] > 0


def test_batch_parallel(tmp_path: Path, csv_trace_path: Path):
    inp = tmp_path / "in"
    out = tmp_path / "out"
    inp.mkdir()
    # Two copies of the synthetic CSV
    for i in range(2):
        pd.read_csv(csv_trace_path).to_csv(inp / f"rec{i}.csv", index=False)
    summary = batch_detect(
        inp,
        out,
        BatchDetectConfig(
            interval_length=1.0,
            fit_dwelltime=False,
            n_jobs=2,
            analyze_levels=True,
        ),
    )
    assert len(summary) == 2
    assert (summary["status"] == "ok").all()
    meta = (out / "run_metadata.json").read_text(encoding="utf-8")
    assert '"n_jobs": 2' in meta
    assert "1.1.0" in meta
