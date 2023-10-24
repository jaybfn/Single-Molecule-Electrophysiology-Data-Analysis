import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os
import pyabf

from even_detection import ReadingData, CreatingChunks, EventDetection, Plotting 

# Function to load and process your data
def process_data(file_path):
    reader = ReadingData(file_path)
    data = reader.get_data()

    all_events = []

    # Check if the data is from an ABF file or a CSV file
    if isinstance(data, pyabf.ABF):
        chunker = CreatingChunks(data)
        detector = EventDetection(std_multiplier=0.5, threshold_multiplier=2.5)

        for sweepNumber in data.sweepList:
            data.setSweep(sweepNumber)
            sweep_data = data.sweepY * (-1)
            sweep_time = data.sweepX

            for chunk_start in range(0, len(sweep_data), chunker.points_per_interval):
                chunk_end = chunk_start + chunker.points_per_interval
                data_chunk = sweep_data[chunk_start:chunk_end]
                time_chunk = sweep_time[chunk_start:chunk_end]
                events_data = detector.detect_events(data_chunk, time_chunk)
                all_events.extend(events_data)
            
        events_df = pd.DataFrame(all_events)
        return sweep_time, sweep_data, all_events, events_df

    elif isinstance(data, pd.DataFrame):
        # If the data is from a CSV file, you'll need to extract and process the data differently.
        # This is just a placeholder, you need to adjust the processing depending on how the CSV is structured.
        # For example:
        sweep_time = data['time_column']  # Adjust 'time_column' to your actual time column name
        sweep_data = data['data_column']*(-1)  # Adjust 'data_column' to your actual data column name

        # Here you should add your logic for event detection based on your CSV data
        # all_events = your_csv_event_detection_logic(sweep_time, sweep_data)

        events_df = pd.DataFrame(all_events)
        return sweep_time, sweep_data, all_events, events_df
    else:
        raise ValueError("Unsupported data type")

# Main function for Streamlit app
def main():
    st.title('ABF File Data Visualization')

    # Folder selector
    folder_path = st.text_input('Enter folder path here:')
    if folder_path:
        files = os.listdir(folder_path)
        selected_file = st.selectbox("Choose a file from the folder:", files)
        if selected_file:
            data_file = os.path.join(folder_path, selected_file)

            try:
                sweep_time, sweep_data, all_events, events_df = process_data(data_file)
                st.write(events_df)  # Display the events data as a table in Streamlit

                # Use your plotter to display the plot
                plotter = Plotting()
                plot_figure = plotter.plot_data(sweep_time, sweep_data, all_events)
                st.plotly_chart(plot_figure)

            except Exception as e:
                st.write("Error: ", e)

if __name__ == "__main__":
    main()
