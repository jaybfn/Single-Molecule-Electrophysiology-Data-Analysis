import pyabf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
import numpy as np
import pandas as pd
import os

class ReadingData:
    """
    A class used to represent the ReadingData process.

    Attributes
    ----------
    data : object
        An object that contains the data from the file, can be a DataFrame or pyabf.ABF object.

    Methods
    -------
    get_data():
        Returns the data object that contains the data from the file.
    """

    def __init__(self, file_path):
        """
        Constructs all the necessary attributes for the ReadingData object.

        Parameters
        ----------
            file_path : str
                The file path of the file to be read.
        """

        # Check the file extension
        _, file_extension = os.path.splitext(file_path)

        if file_extension.lower() == ".abf":
            # Load the ABF file
            self.data = pyabf.ABF(file_path)
        elif file_extension.lower() == ".csv":
            # Load the CSV file
            self.data = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def get_data(self):
        """
        Retrieves the data loaded during the initialization of the object.

        Returns
        -------
        object
            The data object containing the data from the file, can be a DataFrame or pyabf.ABF object.
        """

        return self.data

class CreatingChunks:
    """
    A class used to handle the creation of data chunks from the continuous sweep data.
    ...

    Attributes
    ----------
    abf : object
        a pyabf.ABF object that contains the data from the ABF file
    points_per_sec : int
        the number of data points recorded per second in the ABF file
    interval_length : int
        the interval length for each chunk in seconds (default is 5 seconds)
    points_per_interval : int
        the total number of data points in each chunk, calculated based on points_per_sec and interval_length

    Methods
    -------
    generate_chunks(sweep_data):
        Generates chunks of sweep data based on the predefined interval length.
    """

    def __init__(self, abf, interval_length=5):
        """
        Constructs all the necessary attributes for the CreatingChunks object.
        Parameters
        ----------
        abf : object
            The pyabf.ABF object that contains the data from the ABF file.
        interval_length : int, optional
            The interval length for each chunk in seconds (default is 5 seconds).
        """

        self.abf = abf
        self.points_per_sec = abf.dataRate  # Determine the number of data points per second
        self.interval_length = interval_length  # Set the interval length for each chunk
        self.points_per_interval = self.points_per_sec * self.interval_length  # Calculate points per interval

    def generate_chunks(self, sweep_data):
        """
        Yields consecutive chunks of data from the sweep_data.
        This generator function divides the continuous data into smaller chunks 
        for more manageable analysis. Each chunk is defined by the interval_length attribute.
        Parameters
        ----------
        sweep_data : ndarray
            An array containing the continuous sweep data.

        Yields
        ------
        ndarray
            A chunk of the sweep data, the size of which is determined by points_per_interval.
        """

        for start in range(0, len(sweep_data), self.points_per_interval):
            end = start + self.points_per_interval  # Determine the end point of the chunk
            yield sweep_data[start:end]  # Yield the chunk of data for further processing


class EventDetection:
    """
    A class used to detect events based on the analysis of data chunks.
    ...

    Attributes
    ----------
    std_multiplier : float
        the multiplier for the standard deviation to calculate the standard deviation threshold
    threshold_multiplier : float
        the multiplier for the standard deviation to calculate the absolute threshold

    Methods
    -------
    detect_events(data_chunk, data_time):
        Analyzes a chunk of data and detects events based on predefined conditions.
    """

    def __init__(self, std_multiplier, threshold_multiplier):
        """
        Constructs all the necessary attributes for the EventDetection object.

        Parameters
        ----------
        std_multiplier : float
            Multiplier for the standard deviation to calculate the standard deviation threshold.
        threshold_multiplier : float
            Multiplier for the standard deviation to calculate the absolute threshold.
        """
        self.std_multiplier = std_multiplier
        self.threshold_multiplier = threshold_multiplier

    def detect_events(self, data_chunk, data_time):
        """
        Detects and records events from a chunk of data based on threshold conditions.
        This method iterates through a data chunk and identifies events where the data points
        cross below the standard deviation threshold and the absolute threshold. Events are recorded
        with details including the event number, start time, end time, and duration.

        Parameters
        ----------
        data_chunk : ndarray
            An array containing a segment of the continuous data.
        data_time : ndarray
            An array containing the time points corresponding to the data_chunk.

        Returns
        -------
        list
            A list of dictionaries, each containing details of an event (event number, start time, end time, and duration).
        """

        events_data = []  # List to store events
        mean = np.mean(data_chunk)  # Calculate the mean of the data chunk
        std_dev = np.std(data_chunk)  # Calculate the standard deviation of the data chunk
        threshold = mean - self.threshold_multiplier * std_dev  # Determine the absolute threshold
        std_threshold = mean - self.std_multiplier * std_dev  # Determine the standard deviation threshold
        start_time = None  # Variable to store the start time of an event
        crossed_threshold = False  # Flag to check if the threshold has been crossed

        for i in range(1, len(data_chunk)):
            # Check if the data point crosses below the standard deviation threshold
            if data_chunk[i] < std_threshold and data_chunk[i - 1] >= std_threshold:
                start_time = data_time[i]  # Record the start time of the event

            # Check if the data point goes below the absolute threshold during the event
            if start_time and data_chunk[i] < threshold:
                crossed_threshold = True  # Set the flag indicating the threshold has been crossed

            # Check if the data point crosses back above the standard deviation threshold
            if start_time and data_chunk[i] >= std_threshold and data_chunk[i - 1] < std_threshold:
                if crossed_threshold:
                    end_time = data_time[i]  # Record the end time of the event
                    difference = end_time - start_time  # Calculate the duration of the event
                    if difference >= 0.0001:  # Check if the event duration meets the minimum criteria
                        # Record the event details
                        events_data.append({
                            'start_time': start_time,
                            'end_time': end_time,
                            'difference': difference
                        })
                        start_time = None  # Reset the start time for the next event
                        crossed_threshold = False  # Reset the threshold flag for the next event

        return events_data  # Return the recorded events


class Plotting:
    """
    A class used for plotting data chunks and events using Plotly.

    ...

    Methods
    -------
    plot_data(data_time, data_chunk, events_data, sigma=1.5):
        Plots the signal, smoothed signal, and events on a graph.
    """

    @staticmethod
    def plot_data(data_time, data_chunk, events_data, sigma=1.5):
        """
        [The same docstring as you've written, with adjustments noting it now uses Plotly and returns a Figure]
        """

        # Convert the numpy arrays to lists for Plotly compatibility
        data_time_list = data_time.tolist()
        data_chunk_list = data_chunk.tolist()

        # Apply Gaussian filter to smooth the data
        smoothed_data = gaussian_filter1d(data_chunk, sigma=sigma)

        # Create figure and add traces
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_time_list, 
                                y=smoothed_data, 
                                mode='lines', 
                                name='Smoothed Signal'))
        fig.add_trace(go.Scatter(x=data_time_list, 
                                y=data_chunk_list, 
                                mode='lines', 
                                name='Signal'))

        # Calculate mean and standard deviation of the data chunk
        mean = np.mean(data_chunk)
        std_dev = np.std(data_chunk)

        # Add horizontal lines for mean and standard deviations
        fig.add_hline(y=mean, 
                    line=dict(color="red", width=2), 
                    name="Mean")
        fig.add_hline(y=mean - 0.25 * std_dev, 
                    line=dict(color="blue", width=2, dash="dash"), 
                    name="0.5x Std Dev")
        fig.add_hline(y=mean - 1.5 * std_dev, 
                    line=dict(color="green", width=2, dash="dot"), 
                    name="2.25x Std Dev")

        # Mark the start and end of events on the plot
        for event in events_data:
            start_time = event['start_time']
            end_time = event['end_time']
            fig.add_trace(go.Scatter(x=[start_time, end_time], 
                                    y=[data_chunk_list[data_time_list.index(start_time)], 
                                    data_chunk_list[data_time_list.index(end_time)]],
                                    mode='markers',
                                    marker=dict(color='white', size=10),
                                    showlegend=False))

        # Update layout
        fig.update_layout(title='Data with Events',
                        xaxis_title='Time (s)',
                        yaxis_title='Current (pA)')

        return fig  # Ensure this method returns the Plotly Figure object
    
    @staticmethod
    def plot_data_series(data_time, data_chunk,sigma=1.5):
        """
        [The same docstring as you've written, with adjustments noting it now uses Plotly and returns a Figure]
        """

        # Convert the numpy arrays to lists for Plotly compatibility
        data_time_list = data_time.tolist()
        data_chunk_list = data_chunk.tolist()

        # Apply Gaussian filter to smooth the data
        smoothed_data = gaussian_filter1d(data_chunk, sigma=sigma)

        # Create figure and add traces
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_time_list, y=smoothed_data, mode='lines', name='Smoothed Signal'))
        fig.add_trace(go.Scatter(x=data_time_list, y=data_chunk_list, mode='lines', name='Signal'))

        # Calculate mean and standard deviation of the data chunk
        mean = np.mean(data_chunk)
        std_dev = np.std(data_chunk)

        # Add horizontal lines for mean and standard deviations
        fig.add_hline(y=mean, line=dict(color="red", width=2), name="Mean")
        fig.add_hline(y=mean - 0.25 * std_dev, line=dict(color="blue", width=2, dash="dash"), name="0.5x Std Dev")
        fig.add_hline(y=mean - 1.5 * std_dev, line=dict(color="green", width=2, dash="dot"), name="2.25x Std Dev")

        # Update layout
        fig.update_layout(title='Ion Current Trace',
                        xaxis_title='Time (s)',
                        yaxis_title='Current (pA)')

        return fig  # Ensure this method returns the Plotly Figure object
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
    print(events_df)

    ind = ((all_events[100]['end_time']+(((all_events[100]['end_time'])/100))*2) * 50000)

    plotter = Plotting()
    plotter.plot_data(sweep_time[:int(ind)], sweep_data[:int(ind)], all_events[:100])
