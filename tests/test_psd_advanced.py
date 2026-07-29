"""Tests for PSD depth features."""

from __future__ import annotations

import numpy as np

from pynanopore.psd.analyzer import PSDAnalyzer
from pynanopore.psd.lorentzian import CompositePSDFitter, LorentzianFitter


def test_compute_psd_custom_window():
    rng = np.random.default_rng(0)
    current = rng.normal(size=4096)
    analyzer = PSDAnalyzer(fs=1000)
    f, p = analyzer.compute_psd(
        current, nperseg=512, noverlap=256, window="hann", scaling="density"
    )
    assert len(f) == len(p) > 0


def test_lorentzian_diagnostics():
    f = np.linspace(1, 2000, 400)
    p = 2.0 / (1.0 + (f / 150.0) ** 2) + 1e-6
    fit = LorentzianFitter(f, p, max_frequency=1500)
    s0, fc = fit.fit_lorentzian()
    assert s0 > 0 and fc > 0
    assert fit.diagnostics is not None
    assert fit.diagnostics.r2_log > 0.9


def test_composite_fit():
    f = np.linspace(1, 2000, 500)
    true = 1.5 / (1.0 + (f / 200.0) ** 2) + 0.05 / (f**1.2)
    fit = CompositePSDFitter(f, true, max_frequency=1500)
    params = fit.fit()
    assert params["S0"] > 0
    assert params["fc"] > 0
    assert params["A"] > 0
    assert fit.diagnostics is not None
    assert fit.diagnostics.n_points > 10
