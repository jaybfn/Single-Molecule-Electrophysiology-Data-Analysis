"""Dwell-time histogram and exponential lifetime fitting (MLE + diagnostics)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import curve_fit, minimize

FitType = Literal["single", "double"]
Binning = Literal["linear", "log"]
FitMethod = Literal["mle", "histogram"]


@dataclass
class DwellTimeFitResult:
    """Result of a dwell-time lifetime fit."""

    fit_type: str
    method: str
    parameters: dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    n_events: int
    bin_centers: list[float]
    hist: list[float]
    fitted: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_dwells(events_df: pd.DataFrame) -> NDArray[np.floating]:
    if "difference" in events_df.columns:
        col = "difference"
    elif "dwell_time" in events_df.columns:
        col = "dwell_time"
    else:
        raise ValueError("events_df must contain 'difference' or 'dwell_time'")
    dwells = events_df[col].to_numpy(dtype=float)
    dwells = dwells[np.isfinite(dwells) & (dwells > 0)]
    if len(dwells) == 0:
        raise ValueError("No positive finite dwell times available")
    return dwells


def _single_pdf(t: NDArray[np.floating], tau: float) -> NDArray[np.floating]:
    tau = max(tau, 1e-15)
    return (1.0 / tau) * np.exp(-t / tau)


def _double_pdf(
    t: NDArray[np.floating], w: float, tau1: float, tau2: float
) -> NDArray[np.floating]:
    w = float(np.clip(w, 1e-9, 1.0 - 1e-9))
    return w * _single_pdf(t, tau1) + (1.0 - w) * _single_pdf(t, tau2)


def _aic(k: int, log_like: float) -> float:
    return 2.0 * k - 2.0 * log_like


def _bic(k: int, n: int, log_like: float) -> float:
    return k * np.log(max(n, 1)) - 2.0 * log_like


class DwellTimeExponentialFit:
    """
    Fit dwell-time distributions.

    Preferred path: :meth:`fit` with ``method='mle'`` returning physical lifetimes ``τ``.
    Legacy histogram ``a * exp(b * x)`` curve_fit remains available via ``fit_data``.
    """

    def __init__(
        self,
        events_df: pd.DataFrame,
        bins: int = 250,
        *,
        binning: Binning = "linear",
    ) -> None:
        if bins < 1:
            raise ValueError("bins must be >= 1")
        self.events_df = events_df
        self.bins = int(bins)
        self.binning: Binning = binning
        self.dwells = _extract_dwells(events_df)
        self.hist, self.bin_centers = self._prepare_histogram()
        self.params_single: NDArray[np.floating] | None = None
        self.params_double: NDArray[np.floating] | None = None
        self.last_result: DwellTimeFitResult | None = None

    def _prepare_histogram(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        dwells = self.dwells
        if self.binning == "log":
            positive = dwells[dwells > 0]
            lo, hi = float(positive.min()), float(positive.max())
            if lo >= hi:
                hi = lo * 1.1 + 1e-12
            edges = np.geomspace(lo, hi, self.bins + 1)
            hist, bins = np.histogram(positive, bins=edges, density=True)
        else:
            hist, bins = np.histogram(dwells, bins=self.bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        return hist.astype(float), bin_centers.astype(float)

    # --- Legacy histogram models (a * exp(b x)) ---------------------------------
    @staticmethod
    def single_exponential(x: NDArray[np.floating], a: float, b: float) -> NDArray[np.floating]:
        return a * np.exp(b * x)

    @staticmethod
    def double_exponential(
        x: NDArray[np.floating], a: float, b: float, c: float, d: float
    ) -> NDArray[np.floating]:
        return a * np.exp(b * x) + c * np.exp(d * x)

    def fit_data(self, fit_type: FitType) -> NDArray[np.floating]:
        """Legacy: fit histogram density with unconstrained exponentials."""
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
        """Return legacy histogram-fit parameters."""
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
                float(self.params_double[3]),
            )
        raise ValueError("fit_type must be either 'single' or 'double'")

    def print_parameters(self, fit_type: FitType) -> tuple[float, ...]:
        return self.get_parameters(fit_type)

    def fitted_curve(self, fit_type: FitType) -> NDArray[np.floating]:
        """Evaluate the active model on histogram bin centers."""
        if self.last_result is not None and self.last_result.fit_type == fit_type:
            return np.asarray(self.last_result.fitted, dtype=float)
        if fit_type == "single":
            if self.params_single is None:
                raise RuntimeError("Call fit_data('single') or fit() first")
            return self.single_exponential(self.bin_centers, *self.params_single)
        if fit_type == "double":
            if self.params_double is None:
                raise RuntimeError("Call fit_data('double') or fit() first")
            return self.double_exponential(self.bin_centers, *self.params_double)
        raise ValueError("fit_type must be either 'single' or 'double'")

    # --- Modern MLE / physical lifetimes ----------------------------------------
    def fit_mle_single(self) -> DwellTimeFitResult:
        """MLE for Exponential(τ): τ̂ = mean(t)."""
        dwells = self.dwells
        tau = float(np.mean(dwells))
        ll = float(np.sum(np.log(_single_pdf(dwells, tau))))
        fitted = _single_pdf(self.bin_centers, tau)
        result = DwellTimeFitResult(
            fit_type="single",
            method="mle",
            parameters={"tau": tau},
            log_likelihood=ll,
            aic=_aic(1, ll),
            bic=_bic(1, len(dwells), ll),
            n_events=len(dwells),
            bin_centers=self.bin_centers.tolist(),
            hist=self.hist.tolist(),
            fitted=fitted.tolist(),
        )
        self.last_result = result
        return result

    def fit_mle_double(self) -> DwellTimeFitResult:
        """MLE for mixture w/τ1 * exp + (1-w)/τ2 * exp."""
        dwells = self.dwells
        mean = float(np.mean(dwells))

        def neg_ll(theta: NDArray[np.floating]) -> float:
            logit_w, log_t1, log_t2 = theta
            w = 1.0 / (1.0 + np.exp(-logit_w))
            tau1 = float(np.exp(log_t1))
            tau2 = float(np.exp(log_t2))
            pdf = _double_pdf(dwells, w, tau1, tau2)
            return -float(np.sum(np.log(np.clip(pdf, 1e-300, None))))

        x0 = np.array([0.0, np.log(mean * 0.5 + 1e-12), np.log(mean * 2.0 + 1e-12)])
        opt = minimize(neg_ll, x0, method="Nelder-Mead")
        logit_w, log_t1, log_t2 = opt.x
        w = float(1.0 / (1.0 + np.exp(-logit_w)))
        tau1 = float(np.exp(log_t1))
        tau2 = float(np.exp(log_t2))
        # Order components so tau1 <= tau2
        if tau1 > tau2:
            tau1, tau2 = tau2, tau1
            w = 1.0 - w

        ll = -float(opt.fun)
        fitted = _double_pdf(self.bin_centers, w, tau1, tau2)
        result = DwellTimeFitResult(
            fit_type="double",
            method="mle",
            parameters={"w": w, "tau1": tau1, "tau2": tau2},
            log_likelihood=ll,
            aic=_aic(3, ll),
            bic=_bic(3, len(dwells), ll),
            n_events=len(dwells),
            bin_centers=self.bin_centers.tolist(),
            hist=self.hist.tolist(),
            fitted=fitted.tolist(),
        )
        self.last_result = result
        return result

    def fit(
        self,
        fit_type: FitType | Literal["auto"] = "single",
        *,
        method: FitMethod = "mle",
    ) -> DwellTimeFitResult:
        """
        Fit dwell times.

        ``fit_type='auto'`` (MLE only) picks single vs double by lower AIC.
        """
        if method == "histogram":
            chosen: FitType = "single" if fit_type == "auto" else fit_type
            self.fit_data(chosen)
            # Map legacy params to approximate tau = -1/b when b < 0
            if chosen == "single":
                a, b = self.get_parameters("single")
                tau = (-1.0 / b) if b < 0 else float("nan")
                fitted = self.fitted_curve("single")
                # Pseudo LL from density at events (rough)
                pdf = np.clip(a * np.exp(b * self.dwells), 1e-300, None)
                ll = float(np.sum(np.log(pdf)))
                result = DwellTimeFitResult(
                    fit_type="single",
                    method="histogram",
                    parameters={"a": float(a), "b": float(b), "tau": float(tau)},
                    log_likelihood=ll,
                    aic=_aic(2, ll),
                    bic=_bic(2, len(self.dwells), ll),
                    n_events=len(self.dwells),
                    bin_centers=self.bin_centers.tolist(),
                    hist=self.hist.tolist(),
                    fitted=fitted.tolist(),
                )
            else:
                a, b, c, d = self.get_parameters("double")
                fitted = self.fitted_curve("double")
                pdf = np.clip(
                    a * np.exp(b * self.dwells) + c * np.exp(d * self.dwells),
                    1e-300,
                    None,
                )
                ll = float(np.sum(np.log(pdf)))
                result = DwellTimeFitResult(
                    fit_type="double",
                    method="histogram",
                    parameters={
                        "a": float(a),
                        "b": float(b),
                        "c": float(c),
                        "d": float(d),
                        "tau1": float(-1.0 / b) if b < 0 else float("nan"),
                        "tau2": float(-1.0 / d) if d < 0 else float("nan"),
                    },
                    log_likelihood=ll,
                    aic=_aic(4, ll),
                    bic=_bic(4, len(self.dwells), ll),
                    n_events=len(self.dwells),
                    bin_centers=self.bin_centers.tolist(),
                    hist=self.hist.tolist(),
                    fitted=fitted.tolist(),
                )
            self.last_result = result
            return result

        # MLE
        if fit_type == "auto":
            single = self.fit_mle_single()
            double = self.fit_mle_double()
            result = single if single.aic <= double.aic else double
            self.last_result = result
            return result
        if fit_type == "single":
            return self.fit_mle_single()
        if fit_type == "double":
            return self.fit_mle_double()
        raise ValueError("fit_type must be 'single', 'double', or 'auto'")

    def compare_models(self) -> dict[str, DwellTimeFitResult]:
        """Return MLE fits for single and double with AIC/BIC."""
        return {
            "single": self.fit_mle_single(),
            "double": self.fit_mle_double(),
        }


DwellTime_ExponentialFit = DwellTimeExponentialFit
