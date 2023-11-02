import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from event_detection import ReadingData, CreatingChunks, EventDetection, Plotting 

class DwellTime_ExponentialFit:
    def __init__(self, events_df: pd.DataFrame, bins: int = 250) -> None:
        """Initialize the ExponentialFit class."""
        self.events_df = events_df
        self.bins = bins
        self.hist, self.bin_centers = self._prepare_histogram()
        self.params_single = None
        self.params_double = None

    def _prepare_histogram(self) -> (np.ndarray, np.ndarray):
        """Prepare histogram for event data."""
        hist, bins = np.histogram(self.events_df.difference, bins=self.bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        return hist, bin_centers

    @staticmethod
    def single_exponential(x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Single exponential function."""
        return a * np.exp(b * x)

    @staticmethod
    def double_exponential(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        """Double exponential function."""
        return a * np.exp(b * x) + c * np.exp(d * x)

    def fit_data(self, fit_type: str) -> None:
        """Fit data to single or double exponential functions based on fit_type."""
        if fit_type == 'single':
            self.params_single, _ = curve_fit(self.single_exponential, self.bin_centers, self.hist)
        elif fit_type == 'double':
            self.params_double, _ = curve_fit(self.double_exponential, self.bin_centers, self.hist)
        else:
            raise ValueError("fit_type must be either 'single' or 'double'")

    def plot_hist_data(self) -> None:
        """Plot the histogram"""
        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Bar(x=self.bin_centers, y=self.hist, name="Histogram"))

        fig.update_layout(
            title="Dwell Time Histogram",
            xaxis_title="Dwell Time (s)",
            yaxis_title="Counts"
        )
        
        return fig

    def plot_data(self, fit_type: str) -> None:
        """Plot the histogram and the fits using Plotly based on fit_type."""
        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Bar(x=self.bin_centers, y=self.hist, name="Histogram"))
        
        if fit_type == 'single':
            # Single Exponential Fit
            fig.add_trace(go.Scatter(x=self.bin_centers, y=self.single_exponential(self.bin_centers, *self.params_single), mode='lines', name='Single Exponential'))
        elif fit_type == 'double':
            # Double Exponential Fit
            fig.add_trace(go.Scatter(x=self.bin_centers, y=self.double_exponential(self.bin_centers, *self.params_double), mode='lines', name='Double Exponential'))
        else:
            raise ValueError("fit_type must be either 'single' or 'double'")
        
        fig.update_layout(
            title="Histogram with Exponential Fit",
            xaxis_title="Dwell Time (s)",
            yaxis_title="Counts"
        )
        
        return fig

    def print_parameters(self, fit_type: str) -> None:
        """Print the fitting parameters based on fit_type."""
        if fit_type == 'single':
            #print(f"Single Exponential Parameters: a = {self.params_single[0]:.4f}, b = {self.params_single[1]:.4f}")
            a = self.params_single[0]
            b = self.params_single[1]
            return a, b
        elif fit_type == 'double':
            #print(f"Double Exponential Parameters: a = {self.params_double[0]:.4f}, b = {self.params_double[1]:.4f}, c = {self.params_double[2]:.4f}, d = {self.params_double[3]:.4f}")
            a = self.params_double[0]
            b = self.params_double[1]
            c = self.params_double[2]
            d = self.params_double[2]
            return a, b, c,d
        else:
            raise ValueError("fit_type must be either 'single' or 'double'")
        

if __name__ == "__main__":

    reader = ReadingData("../data/2019_04_03_0006.abf")
    abf = reader.get_data()

    chunker = CreatingChunks(abf)
    detector = EventDetection(std_multiplier=0.25, threshold_multiplier=1.5)

    all_events = []

    for sweepNumber in abf.sweepList:
        abf.setSweep(sweepNumber)
        sweep_data = abf.sweepY * (-1)
        sweep_time = abf.sweepX

        for chunk_start in range(0, len(sweep_data), chunker.points_per_interval):
            chunk_end = chunk_start + chunker.points_per_interval
            data_chunk = sweep_data[chunk_start:chunk_end]
            time_chunk = sweep_time[chunk_start:chunk_end]
            events_data = detector.detect_events(data_chunk, time_chunk)
            all_events.extend(events_data)

    events_df = pd.DataFrame(all_events)
    #print(events_df)

    fit = DwellTime_ExponentialFit(events_df)
    fit.fit_data('single')
    fit.plot_hist_data()
    fit.plot_data('single')
    fit.print_parameters('single')