"""Welch PSD estimation for ion-current signals."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch


class PSDAnalyzer:
    """Compute one-sided power spectral density via Welch's method."""

    def __init__(self, fs: float = 50000.0):
        if fs <= 0:
            raise ValueError("fs must be positive")
        self.fs = float(fs)

    def compute_psd_with_hamming(
        self,
        current_data: NDArray[np.floating],
        nperseg: int | None = None,
        noverlap: int | None = None,
        *,
        skip_bins: int = 2,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Compute PSD using a Hamming window.

        Returns
        -------
        frequencies, power_spectrum
        """
        if len(current_data) < 4:
            raise ValueError("current_data too short for PSD estimation")

        if nperseg is None:
            nperseg = max(4, len(current_data) // 2)
        if noverlap is None:
            noverlap = nperseg // 4

        frequencies, power_spectrum = welch(
            current_data,
            self.fs,
            window="hamming",
            nperseg=nperseg,
            noverlap=noverlap,
            return_onesided=True,
            scaling="spectrum",
        )

        if skip_bins > 0:
            frequencies = frequencies[skip_bins:]
            power_spectrum = power_spectrum[skip_bins:]

        return frequencies.astype(float), power_spectrum.astype(float)
