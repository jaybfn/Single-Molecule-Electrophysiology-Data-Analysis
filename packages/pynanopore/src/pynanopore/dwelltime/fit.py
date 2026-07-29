"""Dwell-time histogram and single/double exponential fits."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import curve_fit

FitType = Literal["single", "double"]


class DwellTimeExponentialFit:
    """Fit dwell-time histograms to single or double exponential models."""

    def __init__(self, events_df: pd.DataFrame, bins: int = 250) -> None:
        if "difference" not in events_df.columns:
            raise ValueError("events_df must contain a 'difference' column")
        if bins < 1:
            raise ValueError("bins must be >= 1")
        self.events_df = events_df
        self.bins = int(bins)
        self.hist, self.bin_centers = self._prepare_histogram()
        self.params_single: NDArray[np.floating] | None = None
        self.params_double: NDArray[np.floating] | None = None

    def _prepare_histogram(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        hist, bins = np.histogram(self.events_df["difference"], bins=self.bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        return hist.astype(float), bin_centers.astype(float)

    @staticmethod
    def single_exponential(x: NDArray[np.floating], a: float, b: float) -> NDArray[np.floating]:
        return a * np.exp(b * x)

    @staticmethod
    def double_exponential(
        x: NDArray[np.floating], a: float, b: float, c: float, d: float
    ) -> NDArray[np.floating]:
        return a * np.exp(b * x) + c * np.exp(d * x)

    def fit_data(self, fit_type: FitType) -> NDArray[np.floating]:
        """Fit histogram density; returns optimized parameters."""
        if fit_type == "single":
            self.params_single, _ = curve_fit(
                self.single_exponential, self.bin_centers, self.hist, maxfev=10000
            )
            return self.params_single
        if fit_type == "double":
            self.params_double, _ = curve_fit(
                self.double_exponential, self.bin_centers, self.hist, maxfev=20000
            )
            return self.params_double
        raise ValueError("fit_type must be either 'single' or 'double'")

    def get_parameters(self, fit_type: FitType) -> tuple[float, ...]:
        """Return fitted parameters for the requested model."""
        if fit_type == "single":
            if self.params_single is None:
                raise RuntimeError("Call fit_data('single') before get_parameters")
            return float(self.params_single[0]), float(self.params_single[1])
        if fit_type == "double":
            if self.params_double is None:
                raise RuntimeError("Call fit_data('double') before get_parameters")
            return (
                float(self.params_double[0]),
                float(self.params_double[1]),
                float(self.params_double[2]),
                float(self.params_double[3]),  # fixed: previously returned [2] twice
            )
        raise ValueError("fit_type must be either 'single' or 'double'")

    # Backward-compatible alias
    def print_parameters(self, fit_type: FitType) -> tuple[float, ...]:
        return self.get_parameters(fit_type)

    def fitted_curve(self, fit_type: FitType) -> NDArray[np.floating]:
        """Evaluate the fitted model on histogram bin centers."""
        if fit_type == "single":
            if self.params_single is None:
                raise RuntimeError("Call fit_data('single') first")
            return self.single_exponential(self.bin_centers, *self.params_single)
        if fit_type == "double":
            if self.params_double is None:
                raise RuntimeError("Call fit_data('double') first")
            return self.double_exponential(self.bin_centers, *self.params_double)
        raise ValueError("fit_type must be either 'single' or 'double'")


# Backward-compatible name
DwellTime_ExponentialFit = DwellTimeExponentialFit
