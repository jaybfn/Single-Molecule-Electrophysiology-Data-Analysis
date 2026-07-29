"""Tests for PSD and Lorentzian fitting."""

from __future__ import annotations

import numpy as np

from pynanopore.psd.analyzer import PSDAnalyzer
from pynanopore.psd.lorentzian import LorentzianFitter


def test_compute_psd():
    rng = np.random.default_rng(0)
    current = rng.normal(size=2000)
    analyzer = PSDAnalyzer(fs=1000)
    frequencies, power_spectrum = analyzer.compute_psd_with_hamming(current)
    assert len(frequencies) > 0
    assert len(frequencies) == len(power_spectrum)


def test_fit_lorentzian():
    frequencies = np.linspace(1, 1000, 200)
    power_spectrum = 1.0 / (1.0 + (frequencies / 100.0) ** 2) + 1e-6
    fitter = LorentzianFitter(frequencies, power_spectrum)
    s0, fc = fitter.fit_lorentzian()
    assert s0 is not None and fc is not None
    assert s0 > 0 and fc > 0
