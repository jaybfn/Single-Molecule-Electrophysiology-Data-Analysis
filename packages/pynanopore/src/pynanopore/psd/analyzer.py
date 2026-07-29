"""Welch PSD estimation for ion-current signals."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch

WindowType = Literal[
    "hamming",
    "hann",
    "boxcar",
    "triang",
    "blackman",
    "bartlett",
    "flattop",
]
ScalingType = Literal["density", "spectrum"]


class PSDAnalyzer:
    """Compute one-sided power spectral density via Welch's method."""

    def __init__(self, fs: float = 50000.0):
        if fs <= 0:
            raise ValueError("fs must be positive")
        self.fs = float(fs)

    def compute_psd(
        self,
        current_data: NDArray[np.floating],
        nperseg: int | None = None,
        noverlap: int | None = None,
        *,
        window: WindowType = "hamming",
        scaling: ScalingType = "spectrum",
        skip_bins: int = 2,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Compute PSD with configurable Welch parameters.

        Returns
        -------
        frequencies, power_spectrum
        """
        if len(current_data) < 4:
            raise ValueError("current_data too short for PSD estimation")

        if nperseg is None:
            nperseg = max(4, len(current_data) // 2)
        nperseg = min(int(nperseg), len(current_data))
        if noverlap is None:
            noverlap = nperseg // 4
        noverlap = min(int(noverlap), nperseg - 1)

        frequencies, power_spectrum = welch(
            current_data,
            self.fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            return_onesided=True,
            scaling=scaling,
        )

        if skip_bins > 0:
            frequencies = frequencies[skip_bins:]
            power_spectrum = power_spectrum[skip_bins:]

        return frequencies.astype(float), power_spectrum.astype(float)

    def compute_psd_with_hamming(
        self,
        current_data: NDArray[np.floating],
        nperseg: int | None = None,
        noverlap: int | None = None,
        *,
        skip_bins: int = 2,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Backward-compatible Hamming / spectrum PSD."""
        return self.compute_psd(
            current_data,
            nperseg=nperseg,
            noverlap=noverlap,
            window="hamming",
            scaling="spectrum",
            skip_bins=skip_bins,
        )
