"""Lorentzian power-1 model fitting for PSD curves."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares


class LorentzianFitter:
    """Fit a Lorentzian S0 / (1 + (f/fc)^2) model on a log-log scale."""

    def __init__(
        self,
        frequencies: NDArray[np.floating],
        power_spectrum: NDArray[np.floating],
        *,
        max_frequency: float = 10000.0,
    ):
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.power_spectrum = np.asarray(power_spectrum, dtype=float)
        self.max_frequency = float(max_frequency)
        self.S_0_opt: float | None = None
        self.f_c_opt: float | None = None
        self.filtered_frequencies: NDArray[np.floating] | None = None
        self.filtered_power_spectrum: NDArray[np.floating] | None = None

    @staticmethod
    def lorentzian_power1(f: NDArray[np.floating] | float, S_0: float, f_c: float):
        return S_0 / (1.0 + (np.asarray(f, dtype=float) / f_c) ** 2)

    def residuals_log(
        self,
        params: list[float] | NDArray[np.floating],
        f_log: NDArray[np.floating],
        y_observed: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        S_0, f_c = float(params[0]), float(params[1])
        y_model = np.log10(self.lorentzian_power1(10**f_log, S_0, f_c))
        return y_observed - y_model

    def fit_lorentzian(self) -> tuple[float, float]:
        """Fit and return ``(S_0, f_c)``."""
        if len(self.frequencies) < 3:
            raise ValueError("Need at least 3 frequency bins to fit")

        mask = (self.frequencies <= self.max_frequency) & (self.frequencies > self.frequencies[1])
        self.filtered_frequencies = self.frequencies[mask]
        self.filtered_power_spectrum = self.power_spectrum[mask]

        if len(self.filtered_frequencies) < 3:
            raise ValueError("Insufficient points after frequency filtering")

        # Avoid non-positive power for log fit
        positive = self.filtered_power_spectrum > 0
        self.filtered_frequencies = self.filtered_frequencies[positive]
        self.filtered_power_spectrum = self.filtered_power_spectrum[positive]
        if len(self.filtered_frequencies) < 3:
            raise ValueError("Insufficient positive PSD points for log-space fit")

        initial_guess = [1e-3, 1e3]
        result = least_squares(
            self.residuals_log,
            initial_guess,
            args=(
                np.log10(self.filtered_frequencies),
                np.log10(self.filtered_power_spectrum),
            ),
            method="trf",
            bounds=([1e-10, 1e-10], [1e7, 1e4]),
            max_nfev=100000,
        )
        self.S_0_opt = float(result.x[0])
        self.f_c_opt = float(result.x[1])
        return self.S_0_opt, self.f_c_opt

    def fitted_curve(self) -> NDArray[np.floating]:
        if self.S_0_opt is None or self.f_c_opt is None or self.filtered_frequencies is None:
            raise RuntimeError("Call fit_lorentzian() first")
        return self.lorentzian_power1(self.filtered_frequencies, self.S_0_opt, self.f_c_opt)
