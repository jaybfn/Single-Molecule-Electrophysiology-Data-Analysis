"""Tests for MLE dwell-time fitting and model selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pynanopore.dwelltime.fit import DwellTimeExponentialFit


def test_mle_single_recovers_tau():
    rng = np.random.default_rng(0)
    tau_true = 0.02
    dwells = rng.exponential(scale=tau_true, size=2000)
    fit = DwellTimeExponentialFit(pd.DataFrame({"difference": dwells}), bins=40)
    result = fit.fit("single", method="mle")
    assert result.parameters["tau"] == pytest.approx(tau_true, rel=0.15)
    assert result.aic < 0 or np.isfinite(result.aic)


def test_mle_auto_prefers_single_for_single_component():
    rng = np.random.default_rng(1)
    dwells = rng.exponential(scale=0.01, size=1500)
    fit = DwellTimeExponentialFit(pd.DataFrame({"difference": dwells}), bins=40)
    result = fit.fit("auto", method="mle")
    assert result.fit_type == "single"


def test_mle_double_has_two_taus():
    rng = np.random.default_rng(2)
    d1 = rng.exponential(scale=0.005, size=800)
    d2 = rng.exponential(scale=0.04, size=800)
    dwells = np.concatenate([d1, d2])
    fit = DwellTimeExponentialFit(pd.DataFrame({"difference": dwells}), bins=50)
    result = fit.fit("double", method="mle")
    assert "tau1" in result.parameters and "tau2" in result.parameters
    assert result.parameters["tau1"] < result.parameters["tau2"]


def test_log_binning():
    rng = np.random.default_rng(3)
    dwells = rng.exponential(scale=0.01, size=500)
    fit = DwellTimeExponentialFit(pd.DataFrame({"difference": dwells}), bins=30, binning="log")
    assert len(fit.bin_centers) == 30
    result = fit.fit("single", method="mle")
    assert result.n_events == 500


def test_compare_models():
    rng = np.random.default_rng(4)
    dwells = rng.exponential(scale=0.01, size=400)
    fit = DwellTimeExponentialFit(pd.DataFrame({"difference": dwells}), bins=25)
    both = fit.compare_models()
    assert "single" in both and "double" in both
    assert both["single"].aic <= both["double"].aic + 50  # soft check
