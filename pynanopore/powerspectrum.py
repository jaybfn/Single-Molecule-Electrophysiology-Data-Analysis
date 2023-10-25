import pyabf
import numpy as np
from scipy.signal import welch
from typing import Tuple, List, Optional
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from typing import Tuple, List


class PSDAnalyzer:
    """
    Class for analyzing the power spectral density (PSD) of a current signal.
    """
    def __init__(self, fs: int = 50000):
        """
        Initialize the PSDAnalyzer with a sampling frequency.

        Parameters:
        - fs (int): The sampling frequency. Default is 50000.
        """
        self.fs = fs

    def compute_psd_with_hamming(self, current_data: np.ndarray, nperseg: Optional[int] = None, noverlap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the one-sided PSD of a current signal using Welch's method with a Hamming window.

        Parameters:
        - current_data (np.ndarray): The current signal data.
        - nperseg (Optional[int]): Length of each segment used in Welch's method. 
                                    If None, defaults to len(current_data)//8.
        - noverlap (Optional[int]): Number of points to overlap between segments. 
                                    If None, defaults to nperseg//2.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: frequencies, power_spectrum.
        """
        # If nperseg or noverlap are not provided, set them to default values
        if nperseg is None:
            nperseg = len(current_data) // 2
        if noverlap is None:
            noverlap = nperseg // 4

        # Use Welch's method to estimate the power spectral density
        frequencies, power_spectrum = welch(current_data, self.fs, window='hamming', 
                                            nperseg=nperseg, noverlap=noverlap, 
                                            return_onesided=True, scaling='spectrum')

        # Exclude the first two bins
        frequencies = frequencies[2:]
        power_spectrum = power_spectrum[2:]

        # Plotting
        self._plot_psd(frequencies, power_spectrum)

        return frequencies, power_spectrum

    def _plot_psd(self, frequencies: np.ndarray, power_spectrum: np.ndarray) -> None:
        """
        Plot the power spectrum on a log-log plot using Plotly.

        Parameters:
        - frequencies (np.ndarray): The frequency values.
        - power_spectrum (np.ndarray): The power spectrum values.
        """
        
        # Create a new figure
        fig = go.Figure()

        # Add line plot for the power spectrum
        fig.add_trace(go.Scatter(x=frequencies, 
                                y=power_spectrum, 
                                mode='lines', 
                                line=dict(color='rgb(0,0,255)')))
        
        # Update the layout
        fig.update_layout(
            title='One-sided Power Spectrum with Hamming Window (Welch, Excluding First 2 Bins)',
            xaxis=dict(type='log', title='Frequency (Hz)', range=[np.log10(frequencies[0]), np.log10(50000)]),
            yaxis=dict(type='log', title='Power Spectrum (pA^2/Hz)'),
            hovermode='closest',
            template='plotly_white',
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            xaxis_gridwidth=1,
            yaxis_gridwidth=1)
        
        # Show the figure
        fig.show()


class LorentzianFitter:
    """
    Class to fit a Lorentzian power-1 model to power spectrum data.

    Attributes:
    frequencies (ndarray): Input frequencies for power spectrum.
    power_spectrum (ndarray): Input power spectrum data.
    S_0_opt (float): Optimized S_0 parameter after fitting.
    f_c_opt (float): Optimized f_c parameter after fitting.
    filtered_frequencies (ndarray): Frequencies filtered for fitting.
    filtered_power_spectrum (ndarray): Power spectrum filtered for fitting.
    """

    def __init__(self, frequencies: np.ndarray, power_spectrum: np.ndarray):
        """
        Initialize the LorentzianFitter with frequencies and power_spectrum data.

        Parameters:
        frequencies (ndarray): The frequencies corresponding to the power spectrum data.
        power_spectrum (ndarray): The power spectrum data to be fitted.
        """
        self.frequencies = frequencies
        self.power_spectrum = power_spectrum
        self.S_0_opt = None
        self.f_c_opt = None
        self.filtered_frequencies = None
        self.filtered_power_spectrum = None

    @staticmethod
    def lorentzian_power1(f: float, S_0: float, f_c: float) -> float:
        """
        Lorentzian power-1 model function.

        Parameters:
        f (float): Frequency.
        S_0 (float): S_0 parameter of the Lorentzian model.
        f_c (float): f_c parameter of the Lorentzian model.

        Returns:
        float: The value of the Lorentzian power-1 model at the given frequency.
        """
        return S_0 / (1 + (f / f_c) ** 2)

    def residuals_log(self, params: List[float], f: np.ndarray, y_observed: np.ndarray) -> np.ndarray:
        """
        Compute residuals for the Lorentzian power-1 model on a log-log scale.

        Parameters:
        params (list): List of parameters [S_0, f_c].
        f (ndarray): Frequencies.
        y_observed (ndarray): Observed y-values.

        Returns:
        ndarray: Residuals between observed and model y-values.
        """
        S_0, f_c = params
        y_model = np.log10(self.lorentzian_power1(10**f, S_0, f_c))
        return y_observed - y_model

    def fit_lorentzian(self):
        """Fit the power spectrum data to the Lorentzian power-1 model."""
        initial_guess_lm = [1e-3, 1e3]
        
        # Filter the data
        self.filtered_frequencies = self.frequencies[(self.frequencies <= 10000) & (self.frequencies > self.frequencies[1])]
        self.filtered_power_spectrum = self.power_spectrum[(self.frequencies <= 10000) & (self.frequencies > self.frequencies[1])]

        # Fit using Trust Region Reflective algorithm
        result_log_constrained = least_squares(self.residuals_log, initial_guess_lm, args=(np.log10(self.filtered_frequencies), np.log10(self.filtered_power_spectrum)), method='trf', bounds=([1e-10, 1e-10], [1e7, 1e4]), max_nfev=100000)
        self.S_0_opt, self.f_c_opt = result_log_constrained.x

    def plot_fit(self, frequencies: np.ndarray, power_spectrum: np.ndarray):
        """
        Plot the power spectrum data and the Lorentzian power-1 fitted curve using Plotly.

        Parameters:
        frequencies (ndarray): Frequencies.
        power_spectrum (ndarray): Power spectrum data.
        """
        lorentzian_power1_curve_optimized_log_constrained = self.lorentzian_power1(self.filtered_frequencies, self.S_0_opt, self.f_c_opt)
        
        # Create a new figure
        fig = go.Figure()

        # Add scatter plot for data
        fig.add_trace(go.Scatter(x=frequencies, 
                                y=power_spectrum, 
                                mode='lines', 
                                line=dict(color='rgb(0,0,255)')))

        # Add line plot for the Lorentzian power-1 fit
        fig.add_trace(go.Scatter(x=self.filtered_frequencies, 
                                y=lorentzian_power1_curve_optimized_log_constrained, 
                                mode='lines', 
                                line=dict(color='rgb(0,225,0)'), 
                                ))
        
        # Update the layout
        fig.update_layout(
            title='Power Spectrum Data with Constrained Optimized Lorentzian Power-1 Fit on Log-Log Scale',
            xaxis=dict(type='log', title='Frequency (Hz)', range=[np.log10(frequencies[2]), np.log10(10000)]),
            yaxis=dict(type='log', title='Amplitude (pA^2/Hz)'),
            hovermode='closest'
        )
        
        # Show the figure
        fig.show()

if __name__ == '__main__':

    abf = pyabf.ABF("../data/2019_04_03_0006.abf")
    time_series = sweep_data = abf.sweepY
    current_data = time_series

    # First, compute the power spectrum
    analyzer = PSDAnalyzer(fs=50000)
    frequencies, power_spectrum = analyzer.compute_psd_with_hamming(current_data)

    # Then, fit the Lorentzian model and plot
    lorentzian_fitter = LorentzianFitter(frequencies, power_spectrum)
    lorentzian_fitter.fit_lorentzian()
    lorentzian_fitter.plot_fit(frequencies, power_spectrum)

    # If you need the optimized values:
    print(lorentzian_fitter.S_0_opt, lorentzian_fitter.f_c_opt)