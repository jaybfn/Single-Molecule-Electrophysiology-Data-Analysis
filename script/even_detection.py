import pyabf
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np
import pandas as pd

class ReadingData:
    """
    A class used to represent the ReadingData process.
    ...

    Attributes
    ----------
    abf : object
        a pyabf.ABF object that contains the data from the ABF file

    Methods
    -------
    get_data():
        Returns the abf object that contains the data from the ABF file.
    """

    def __init__(self, file_path):
        """
        Constructs all the necessary attributes for the ReadingData object.
        Parameters
        ----------
            file_path : str
                The file path of the ABF file to be read.
        """

        self.abf = pyabf.ABF(file_path)  # Load the ABF file

    def get_data(self):
        """
        Retrieves the ABF data loaded during the initialization of the object.
        Returns
        -------
        object
            a pyabf.ABF object containing the data from the ABF file
        """

        return self.abf  # Return the ABF data object

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
    A class used for plotting data chunks and events.

    ...

    Methods
    -------
    plot_data(data_time, data_chunk, events_data, sigma=1.5):
        Plots the signal, smoothed signal, and events on a graph.
    """

    @staticmethod
    def plot_data(data_time, data_chunk, events_data, sigma=1.5):
        """
        Plots the data chunk, smoothed data, mean, standard deviation, and events.

        This static method takes the time and amplitude of the data points in a chunk,
        along with detected events, and plots them. It shows the original signal, a smoothed version,
        the mean, specific multiples of the standard deviation, and marks the start and end of events.

        Parameters
        ----------
        data_time : ndarray
            An array containing the time points corresponding to the data_chunk.
        data_chunk : ndarray
            An array containing a segment of the continuous data.
        events_data : list
            A list of dictionaries, each containing details of an event (event number, start time, end time, and duration).
        sigma : float, optional
            The standard deviation for the Gaussian kernel used in smoothing (default is 1.5).

        Returns
        -------
        None
        """

        # Apply Gaussian filter to smooth the data
        smoothed_data = gaussian_filter1d(data_chunk, sigma=sigma)

        # Plot the original signal and the smoothed signal
        plt.plot(data_time, smoothed_data, label="Smoothed Signal")
        plt.plot(data_time, data_chunk, label="Signal")

        # Calculate mean and standard deviation of the data chunk
        mean = np.mean(data_chunk)
        std_dev = np.std(data_chunk)

        # Plot the mean, 0.5x standard deviation, and 2.25x standard deviation
        plt.axhline(y=mean, color='r', linestyle='-', label="Mean")
        plt.axhline(y=mean - 0.5 * std_dev, color='b', linestyle='-', label="0.5x Std Dev")
        plt.axhline(y=mean - 2.25 * std_dev, color='g', linestyle='--', label="2.25x Std Dev")

        # Mark the start and end of events on the plot
        for event in events_data:
            start_time = event['start_time']
            end_time = event['end_time']
            # Convert data_time array to list for indexing
            time_list = data_time.tolist()
            # Plot black circles at the start and end times of the events
            plt.plot(start_time, data_chunk[time_list.index(start_time)], 'ko')
            plt.plot(end_time, data_chunk[time_list.index(end_time)], 'ko')

        # Display the legend and show the plot
        plt.legend()
        plt.show()


if __name__ == "__main__":
    reader = ReadingData("../data/2019_04_03_0006.abf")
    abf = reader.get_data()

    chunker = CreatingChunks(abf)
    detector = EventDetection(std_multiplier=0.5, threshold_multiplier=2.5)

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

    plotter = Plotting()
    plotter.plot_data(sweep_time, sweep_data, all_events)
