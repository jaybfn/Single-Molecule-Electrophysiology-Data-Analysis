"""Lorentzian and composite PSD model fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares


@dataclass
class PSDFitDiagnostics:
    """Goodness-of-fit metrics in log10 space."""

    r2_log: float
    rmse_log: float
    n_points: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _filter_psd(
    frequencies: NDArray[np.floating],
    power_spectrum: NDArray[np.floating],
    *,
    max_frequency: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    if len(frequencies) < 3:
        raise ValueError("Need at least 3 frequency bins to fit")
    mask = (frequencies <= max_frequency) & (frequencies > frequencies[1])
    f = frequencies[mask]
    p = power_spectrum[mask]
    positive = p > 0
    f = f[positive]
    p = p[positive]
    if len(f) < 3:
        raise ValueError("Insufficient positive PSD points for log-space fit")
    return f.astype(float), p.astype(float)


def _log_r2(y_obs: NDArray[np.floating], y_model: NDArray[np.floating]) -> tuple[float, float]:
    resid = y_obs - y_model
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(resid**2)))
    return r2, rmse


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
        self.diagnostics: PSDFitDiagnostics | None = None

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
        self.filtered_frequencies, self.filtered_power_spectrum = _filter_psd(
            self.frequencies, self.power_spectrum, max_frequency=self.max_frequency
        )

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
        y_obs = np.log10(self.filtered_power_spectrum)
        y_model = np.log10(
            self.lorentzian_power1(self.filtered_frequencies, self.S_0_opt, self.f_c_opt)
        )
        r2, rmse = _log_r2(y_obs, y_model)
        self.diagnostics = PSDFitDiagnostics(
            r2_log=r2, rmse_log=rmse, n_points=len(self.filtered_frequencies)
        )
        return self.S_0_opt, self.f_c_opt

    def fitted_curve(self) -> NDArray[np.floating]:
        if self.S_0_opt is None or self.f_c_opt is None or self.filtered_frequencies is None:
            raise RuntimeError("Call fit_lorentzian() first")
        return self.lorentzian_power1(self.filtered_frequencies, self.S_0_opt, self.f_c_opt)


class CompositePSDFitter:
    """
    Fit Lorentzian + power-law (1/f^α) composite model:

        S(f) = S0 / (1 + (f/fc)^2) + A / f^α
    """

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
        self.A_opt: float | None = None
        self.alpha_opt: float | None = None
        self.filtered_frequencies: NDArray[np.floating] | None = None
        self.filtered_power_spectrum: NDArray[np.floating] | None = None
        self.diagnostics: PSDFitDiagnostics | None = None

    @staticmethod
    def model(
        f: NDArray[np.floating],
        S_0: float,
        f_c: float,
        A: float,
        alpha: float,
    ) -> NDArray[np.floating]:
        f = np.asarray(f, dtype=float)
        lor = S_0 / (1.0 + (f / f_c) ** 2)
        powerlaw = A / np.power(np.clip(f, 1e-12, None), alpha)
        return lor + powerlaw

    def residuals_log(
        self,
        params: NDArray[np.floating],
        f_log: NDArray[np.floating],
        y_observed: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        S_0, f_c, A, alpha = (float(x) for x in params)
        y_model = np.log10(self.model(10**f_log, S_0, f_c, A, alpha))
        return y_observed - y_model

    def fit(self) -> dict[str, float]:
        self.filtered_frequencies, self.filtered_power_spectrum = _filter_psd(
            self.frequencies, self.power_spectrum, max_frequency=self.max_frequency
        )
        # Initial guesses
        s0_guess = float(
            np.median(
                self.filtered_power_spectrum[: max(3, len(self.filtered_power_spectrum) // 10)]
            )
        )
        initial = np.array([max(s0_guess, 1e-6), 1e3, max(s0_guess * 0.1, 1e-8), 1.0])
        result = least_squares(
            self.residuals_log,
            initial,
            args=(
                np.log10(self.filtered_frequencies),
                np.log10(self.filtered_power_spectrum),
            ),
            method="trf",
            bounds=([1e-12, 1e-2, 1e-14, 0.0], [1e7, 1e5, 1e3, 3.0]),
            max_nfev=100000,
        )
        self.S_0_opt, self.f_c_opt, self.A_opt, self.alpha_opt = (float(x) for x in result.x)
        y_obs = np.log10(self.filtered_power_spectrum)
        y_model = np.log10(
            self.model(
                self.filtered_frequencies,
                self.S_0_opt,
                self.f_c_opt,
                self.A_opt,
                self.alpha_opt,
            )
        )
        r2, rmse = _log_r2(y_obs, y_model)
        self.diagnostics = PSDFitDiagnostics(
            r2_log=r2, rmse_log=rmse, n_points=len(self.filtered_frequencies)
        )
        return {
            "S0": self.S_0_opt,
            "fc": self.f_c_opt,
            "A": self.A_opt,
            "alpha": self.alpha_opt,
        }

    def fitted_curve(self) -> NDArray[np.floating]:
        if None in (self.S_0_opt, self.f_c_opt, self.A_opt, self.alpha_opt):
            raise RuntimeError("Call fit() first")
        assert self.filtered_frequencies is not None
        return self.model(
            self.filtered_frequencies,
            self.S_0_opt,  # type: ignore[arg-type]
            self.f_c_opt,  # type: ignore[arg-type]
            self.A_opt,  # type: ignore[arg-type]
            self.alpha_opt,  # type: ignore[arg-type]
        )


class LorentzianWhiteFitter:
    """
    Lorentzian plus white-noise floor:

        S(f) = S0 / (1 + (f/fc)^2) + N
    """

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
        self.N_opt: float | None = None
        self.filtered_frequencies: NDArray[np.floating] | None = None
        self.filtered_power_spectrum: NDArray[np.floating] | None = None
        self.diagnostics: PSDFitDiagnostics | None = None

    @staticmethod
    def model(f: NDArray[np.floating], S_0: float, f_c: float, N: float) -> NDArray[np.floating]:
        f = np.asarray(f, dtype=float)
        return S_0 / (1.0 + (f / f_c) ** 2) + N

    def residuals_log(
        self,
        params: NDArray[np.floating],
        f_log: NDArray[np.floating],
        y_observed: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        S_0, f_c, N = (float(x) for x in params)
        y_model = np.log10(np.clip(self.model(10**f_log, S_0, f_c, N), 1e-30, None))
        return y_observed - y_model

    def fit(self) -> dict[str, float]:
        self.filtered_frequencies, self.filtered_power_spectrum = _filter_psd(
            self.frequencies, self.power_spectrum, max_frequency=self.max_frequency
        )
        s0_guess = float(np.median(self.filtered_power_spectrum[: max(3, len(self.filtered_power_spectrum) // 10)]))
        n_guess = float(np.percentile(self.filtered_power_spectrum, 10))
        initial = np.array([max(s0_guess, 1e-6), 1e3, max(n_guess, 1e-12)])
        result = least_squares(
            self.residuals_log,
            initial,
            args=(
                np.log10(self.filtered_frequencies),
                np.log10(self.filtered_power_spectrum),
            ),
            method="trf",
            bounds=([1e-12, 1e-2, 1e-18], [1e7, 1e5, 1e3]),
            max_nfev=100000,
        )
        self.S_0_opt, self.f_c_opt, self.N_opt = (float(x) for x in result.x)
        y_obs = np.log10(self.filtered_power_spectrum)
        y_model = np.log10(
            np.clip(
                self.model(self.filtered_frequencies, self.S_0_opt, self.f_c_opt, self.N_opt),
                1e-30,
                None,
            )
        )
        r2, rmse = _log_r2(y_obs, y_model)
        self.diagnostics = PSDFitDiagnostics(
            r2_log=r2, rmse_log=rmse, n_points=len(self.filtered_frequencies)
        )
        return {"S0": self.S_0_opt, "fc": self.f_c_opt, "N": self.N_opt}

    def fitted_curve(self) -> NDArray[np.floating]:
        if None in (self.S_0_opt, self.f_c_opt, self.N_opt):
            raise RuntimeError("Call fit() first")
        assert self.filtered_frequencies is not None
        return self.model(
            self.filtered_frequencies,
            self.S_0_opt,  # type: ignore[arg-type]
            self.f_c_opt,  # type: ignore[arg-type]
            self.N_opt,  # type: ignore[arg-type]
        )


class MultiLorentzianFitter:
    """
    Sum of ``n`` Lorentzians plus optional white floor:

        S(f) = Σ_i S0_i / (1 + (f/fc_i)^2) + N
    """

    def __init__(
        self,
        frequencies: NDArray[np.floating],
        power_spectrum: NDArray[np.floating],
        *,
        n_components: int = 2,
        include_white: bool = True,
        max_frequency: float = 10000.0,
    ):
        if n_components < 1 or n_components > 3:
            raise ValueError("n_components must be in 1..3")
        self.n_components = int(n_components)
        self.include_white = bool(include_white)
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.power_spectrum = np.asarray(power_spectrum, dtype=float)
        self.max_frequency = float(max_frequency)
        self.params: dict[str, float] = {}
        self.filtered_frequencies: NDArray[np.floating] | None = None
        self.filtered_power_spectrum: NDArray[np.floating] | None = None
        self.diagnostics: PSDFitDiagnostics | None = None
        # Convenience mirrors for first component / noise
        self.S_0_opt: float | None = None
        self.f_c_opt: float | None = None
        self.N_opt: float | None = None

    def model(self, f: NDArray[np.floating], theta: NDArray[np.floating]) -> NDArray[np.floating]:
        f = np.asarray(f, dtype=float)
        out = np.zeros_like(f, dtype=float)
        for i in range(self.n_components):
            s0 = float(theta[2 * i])
            fc = float(theta[2 * i + 1])
            out = out + s0 / (1.0 + (f / fc) ** 2)
        if self.include_white:
            out = out + float(theta[-1])
        return out

    def residuals_log(
        self,
        params: NDArray[np.floating],
        f_log: NDArray[np.floating],
        y_observed: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        y_model = np.log10(np.clip(self.model(10**f_log, params), 1e-30, None))
        return y_observed - y_model

    def fit(self) -> dict[str, float]:
        self.filtered_frequencies, self.filtered_power_spectrum = _filter_psd(
            self.frequencies, self.power_spectrum, max_frequency=self.max_frequency
        )
        s0_guess = float(
            np.median(self.filtered_power_spectrum[: max(3, len(self.filtered_power_spectrum) // 10)])
        )
        f_lo = float(self.filtered_frequencies[0])
        f_hi = float(self.filtered_frequencies[-1])
        initial: list[float] = []
        lo: list[float] = []
        hi: list[float] = []
        for i in range(self.n_components):
            frac = (i + 1) / (self.n_components + 1)
            fc_guess = f_lo * (f_hi / max(f_lo, 1e-12)) ** frac
            initial.extend([max(s0_guess / (i + 1), 1e-6), max(fc_guess, 1.0)])
            lo.extend([1e-12, 1e-2])
            hi.extend([1e7, 1e5])
        if self.include_white:
            n_guess = float(np.percentile(self.filtered_power_spectrum, 10))
            initial.append(max(n_guess, 1e-12))
            lo.append(1e-18)
            hi.append(1e3)

        result = least_squares(
            self.residuals_log,
            np.array(initial),
            args=(
                np.log10(self.filtered_frequencies),
                np.log10(self.filtered_power_spectrum),
            ),
            method="trf",
            bounds=(lo, hi),
            max_nfev=100000,
        )
        theta = result.x
        params: dict[str, float] = {}
        for i in range(self.n_components):
            params[f"S0_{i + 1}"] = float(theta[2 * i])
            params[f"fc_{i + 1}"] = float(theta[2 * i + 1])
        if self.include_white:
            params["N"] = float(theta[-1])
            self.N_opt = params["N"]
        self.S_0_opt = params.get("S0_1")
        self.f_c_opt = params.get("fc_1")
        self.params = params

        y_obs = np.log10(self.filtered_power_spectrum)
        y_model = np.log10(np.clip(self.model(self.filtered_frequencies, theta), 1e-30, None))
        r2, rmse = _log_r2(y_obs, y_model)
        self.diagnostics = PSDFitDiagnostics(
            r2_log=r2, rmse_log=rmse, n_points=len(self.filtered_frequencies)
        )
        return params

    def fitted_curve(self) -> NDArray[np.floating]:
        if not self.params or self.filtered_frequencies is None:
            raise RuntimeError("Call fit() first")
        theta: list[float] = []
        for i in range(self.n_components):
            theta.extend([self.params[f"S0_{i + 1}"], self.params[f"fc_{i + 1}"]])
        if self.include_white:
            theta.append(self.params["N"])
        return self.model(self.filtered_frequencies, np.array(theta))
