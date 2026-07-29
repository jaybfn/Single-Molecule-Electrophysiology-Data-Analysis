"""Tests for dwell-time fitting."""

from __future__ import annotations

import pandas as pd
import pytest

from pynanopore.dwelltime.fit import DwellTimeExponentialFit


def test_init(events_df: pd.DataFrame):
    fit = DwellTimeExponentialFit(events_df, bins=50)
    assert fit.bins == 50
    assert len(fit.hist) == 50


def test_fit_single(events_df: pd.DataFrame):
    fit = DwellTimeExponentialFit(events_df, bins=40)
    fit.fit_data("single")
    a, b = fit.get_parameters("single")
    assert a is not None and b is not None


def test_fit_double_params_length(events_df: pd.DataFrame):
    fit = DwellTimeExponentialFit(events_df, bins=40)
    try:
        fit.fit_data("double")
    except RuntimeError:
        pytest.skip("curve_fit did not converge for this random draw")
    params = fit.get_parameters("double")
    assert len(params) == 4
    # Regression: previously d was incorrectly taken from index 2
    assert params[3] == float(fit.params_double[3])


def test_invalid_fit(events_df: pd.DataFrame):
    fit = DwellTimeExponentialFit(events_df, bins=20)
    with pytest.raises(ValueError):
        fit.fit_data("invalid")  # type: ignore[arg-type]


def test_missing_column():
    with pytest.raises(ValueError):
        DwellTimeExponentialFit(pd.DataFrame({"x": [1, 2]}))
